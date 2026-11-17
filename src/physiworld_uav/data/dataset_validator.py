"""Dataset validation (plan §6.5-DS9). Run 3 scope: integrity, metadata,
designation and leakage checks. Physics-consistency checks (kinematics,
Newton residual, IMU consistency) are added in Run 5.

A dataset is marked ``complete`` in ``index.json`` only if every *hard*
check passes (``validate_dataset(...).ok``).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from physiworld_uav.wind.regimes import Limits, check_limits, realised_stats

from .metadata import EpisodeMetadata
from .schema import BY_PATH, ID_ROLES, validate_arrays

# NaN means "not commanded" (schema semantics): command channels a mode does not use and
# control diagnostics (e.g. no acceleration reference in attitude mode). IMU is gated by imu_valid.
NAN_ALLOWED = ("actions/collective_thrust_cmd", "actions/body_rate_cmd",
               "observations/imu_accel_B", "observations/imu_gyro_B")
NAN_ALLOWED_PREFIXES = ("control/",)
MIN_IMU_VALID_FRACTION = 0.99
REQUIRED_SEEDS = ("episode", "initial_conditions", "trajectory", "wind", "excitation")

#: OOD category -> wind regime categories / domain regions allowed for that role.
ROLE_WIND = {"id": ("id",), "magnitude": ("ood_magnitude",), "direction": ("ood_direction",),
             "temporal": ("ood_temporal",), "gust": ("ood_gust",), "physics": ("id",),
             "trajectory": ("id",), "joint": None}
ROLE_DOMAIN = {"id": ("nominal", "id"), "magnitude": ("nominal", "id"), "direction": ("nominal", "id"),
               "temporal": ("nominal", "id"), "gust": ("nominal", "id"),
               "physics": ("ood_physics", "ood_combination"), "trajectory": ("nominal", "id"),
               "joint": None}


@dataclass
class ValidationReport:
    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    n_episodes: int = 0

    @property
    def ok(self) -> bool:
        return not self.hard

    def merge(self, other: "ValidationReport") -> None:
        self.hard += other.hard
        self.soft += other.soft
        self.n_episodes += other.n_episodes

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "n_episodes": self.n_episodes, "hard": self.hard, "soft": self.soft}


# ---------------------------------------------------------------------------
# Episode-level
# ---------------------------------------------------------------------------


def validate_episode(arrays: Mapping[str, np.ndarray], meta: EpisodeMetadata,
                     *, quat_tol: float = 1e-4, time_tol: float = 1e-9) -> ValidationReport:
    r = ValidationReport(n_episodes=1)
    eid = meta.episode_id
    r.hard += [f"{eid}: {i}" for i in validate_arrays(arrays, data_type=meta.data_type)]
    if r.hard:
        return r

    # finite values
    for path, arr in arrays.items():
        if BY_PATH[path].dtype.startswith("float") and path not in NAN_ALLOWED \
                and not path.startswith(NAN_ALLOWED_PREFIXES):
            if np.isinf(arr).any() or not np.all(np.isfinite(arr)):
                r.hard.append(f"{eid}: non-finite values in {path}")

    for path, arr in arrays.items():                     # diagnostics may be NaN, never ±inf
        if path.startswith(NAN_ALLOWED_PREFIXES) and np.isinf(arr).any():
            r.hard.append(f"{eid}: infinite values in {path}")

    # IMU: observations are model inputs (D7); stale/missing frames are flagged invalid
    valid = arrays["observations/imu_valid"]
    frac = float(valid.mean()) if valid.size else 0.0
    if frac < MIN_IMU_VALID_FRACTION:
        r.hard.append(f"{eid}: IMU delivered fresh frames for only {100 * frac:.1f}% of samples")
    imu = np.concatenate([arrays["observations/imu_accel_B"], arrays["observations/imu_gyro_B"]], axis=1)
    if valid.any() and not np.all(np.isfinite(imu[valid])):
        r.hard.append(f"{eid}: non-finite IMU values in samples flagged valid")

    # quaternions
    q = arrays["ground_truth/orientation_wxyz"].astype(np.float64)
    norm_err = float(np.abs(np.linalg.norm(q, axis=1) - 1.0).max())
    if norm_err > quat_tol:
        r.hard.append(f"{eid}: quaternion norm error {norm_err:.2e} > {quat_tol}")
    if q.shape[0] > 1 and np.any((q[1:] * q[:-1]).sum(axis=1) < 0):
        r.hard.append(f"{eid}: quaternion sign discontinuity")

    # timestamps
    t = arrays["time/t"]
    dt = float(meta.sim.get("physics_dt", np.nan))
    d = np.diff(t)
    if not np.all(d > 0):
        r.hard.append(f"{eid}: timestamps not strictly increasing")
    elif math.isfinite(dt) and np.abs(d - dt).max() > time_tol + 1e-12 * t.max():
        r.hard.append(f"{eid}: non-uniform timestep (max |Δt − dt| = {np.abs(d - dt).max():.2e})")
    dec = int(meta.sim.get("control_decimation", 1))
    if (t.shape[0] - 1) % dec:
        r.hard.append(f"{eid}: {t.shape[0] - 1} steps not a multiple of decimation {dec}")
    if meta.n_steps != t.shape[0] - 1:
        r.hard.append(f"{eid}: metadata n_steps {meta.n_steps} != {t.shape[0] - 1}")

    # metadata completeness
    for k in REQUIRED_SEEDS:
        if k not in meta.seeds:
            r.hard.append(f"{eid}: missing seed '{k}'")
    if not meta.config_hash:
        r.hard.append(f"{eid}: missing config_hash")
    for k in ("mass_kg", "inertia_diag_kgm2", "force_constants"):
        if k not in meta.drone:
            r.hard.append(f"{eid}: drone parameters missing '{k}'")
    for k in ("physics_dt", "control_decimation", "model_dt"):
        if k not in meta.sim:
            r.hard.append(f"{eid}: sim settings missing '{k}'")
    if meta.drone_realised is None:
        r.hard.append(f"{eid}: drone parameters not read back from PhysX")
    if meta.forces.get("applied") and not meta.forces.get("system", {}).get("validated", False):
        r.hard.append(f"{eid}: generated with an unvalidated force config ({meta.forces.get('name')})")
    if not meta.stats.get("applicability_ok", True):
        r.soft.append(f"{eid}: force model outside its applicability envelope "
                      f"{meta.stats.get('applicability_violation_frac')} (excluded from model-ready data)")

    # force accounting: the logged total equals the sum of the logged components (plan F9)
    comp_f = [v for k, v in arrays.items() if k.startswith("ground_truth/wrench/") and k.endswith("_force_W")
              and not k.endswith("external_force_W")]
    comp_t = [v for k, v in arrays.items() if k.startswith("ground_truth/wrench/") and k.endswith("_torque_B")
              and not k.endswith("external_torque_B")]
    tot_f = arrays["ground_truth/wrench/external_force_W"].astype(np.float64)
    tot_t = arrays["ground_truth/wrench/external_torque_B"].astype(np.float64)
    sum_f = np.sum(comp_f, axis=0) if comp_f else np.zeros_like(tot_f)
    sum_t = np.sum(comp_t, axis=0) if comp_t else np.zeros_like(tot_t)
    if np.abs(tot_f - sum_f).max() > 1e-4 or np.abs(tot_t - sum_t).max() > 1e-5:
        r.hard.append(f"{eid}: logged external wrench != sum of logged components "
                      f"(max |Δf| {np.abs(tot_f - sum_f).max():.2e})")

    # wind designation: realised (logged) statistics must equal the planned ones
    wind = meta.plan.get("wind")
    if wind:
        realised = realised_stats(arrays["environment/wind_W"].astype(np.float64))
        planned = wind.get("stats", {})
        n_plan = int(round(float(meta.plan.get("duration_s", np.nan)) / dt)) if math.isfinite(dt) else -1
        complete = meta.status == "completed" and n_plan == meta.n_steps
        if complete:
            for key in ("peak_speed", "mean_horizontal_speed", "max_abs_vertical"):
                if key in planned and abs(realised[key] - planned[key]) > 1e-3:
                    r.hard.append(f"{eid}: logged wind {key} {realised[key]:.4f} != planned {planned[key]:.4f}")
        if wind.get("limits"):
            bad = check_limits(realised, Limits.from_dict(wind["limits"]))
            if bad:
                r.hard.append(f"{eid}: logged wind violates regime {wind.get('regime')}: {bad}")
    return r


# ---------------------------------------------------------------------------
# Dataset-level
# ---------------------------------------------------------------------------


def _role_key(meta: EpisodeMetadata) -> str:
    return "id" if meta.split_role in ID_ROLES else (meta.ood_category or "physics_validation")


def validate_dataset(metas: Sequence[EpisodeMetadata]) -> ValidationReport:
    r = ValidationReport()
    ids = [m.episode_id for m in metas]
    dup = {e for e in ids if ids.count(e) > 1}
    if dup:
        r.hard.append(f"duplicate episode ids: {sorted(dup)[:5]}")

    def one_role(key: str, value_of) -> None:
        roles: dict[Any, set[str]] = defaultdict(set)
        for m in metas:
            v = value_of(m)
            if v is not None:
                roles[v].add(m.split_role)
        leaks = {k: sorted(v) for k, v in roles.items() if len(v) > 1}
        if leaks:
            first = list(leaks.items())[:3]
            r.hard.append(f"{key} shared across split roles ({len(leaks)}): {first}")

    one_role("group_id", lambda m: m.group_id)
    one_role("scenario_id", lambda m: m.scenario_id)
    one_role("counterfactual group", lambda m: m.counterfactual.group_id if m.counterfactual else None)
    one_role("trajectory seed", lambda m: (m.seeds.get("trajectory") or {}).get("seed"))
    one_role("domain id", lambda m: None if m.domain_id == "nominal" else m.domain_id)

    for m in metas:
        role = _role_key(m)
        if role == "physics_validation":
            continue
        wind_cat = m.plan.get("wind", {}).get("category")
        allowed_w = ROLE_WIND.get(role)
        if allowed_w is not None and wind_cat is not None and wind_cat not in allowed_w:
            r.hard.append(f"{m.episode_id}: wind category {wind_cat!r} not allowed in role {m.split_role}")
        region = m.plan.get("domain", {}).get("region")
        allowed_d = ROLE_DOMAIN.get(role)
        if allowed_d is not None and region is not None and region not in allowed_d:
            r.hard.append(f"{m.episode_id}: domain region {region!r} not allowed in role {m.split_role}")

    groups: dict[str, list[EpisodeMetadata]] = defaultdict(list)
    for m in metas:
        if m.counterfactual:
            groups[m.counterfactual.group_id].append(m)
    for gid, members in groups.items():
        base = [m for m in members if m.counterfactual.is_baseline]
        if len(base) != 1:
            r.hard.append(f"counterfactual group {gid}: {len(base)} baselines (need 1)")
        variables = {m.counterfactual.variable for m in members}
        if len(variables) != 1:
            r.hard.append(f"counterfactual group {gid}: mixed variables {sorted(variables)}")
    return r


def validate_all(episodes: Iterable[tuple[Mapping[str, np.ndarray], EpisodeMetadata]]) -> ValidationReport:
    total = ValidationReport()
    metas = []
    for arrays, meta in episodes:
        total.merge(validate_episode(arrays, meta))
        metas.append(meta)
    total.merge(validate_dataset(metas))
    total.n_episodes = len(metas)
    return total
