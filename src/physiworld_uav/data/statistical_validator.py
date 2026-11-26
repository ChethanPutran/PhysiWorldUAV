"""Statistical, physical and coverage validation of a generated dataset (the e2e "data quality" stage).

Complements the schema/integrity ``dataset_validator``: that one asks "is every episode well formed?", this one asks
"is the *dataset* suitable for training?": ranges, intended-vs-realised distributions, duplicates, correlations,
kinematic/physical consistency and coverage of the factors the model must generalise over.

    python -m physiworld_uav.data.statistical_validator --dataset data/datasets/physiworld_pilot_v0.5.0

Writes ``<out>/quality.json``, ``quality_report.md`` and PNG figures (default ``outputs/reports/<dataset>_quality``).
Every check returns findings with a level: ``fail`` (the data is wrong), ``warn`` (look at it), ``info``, ``pass``.
Thresholds are stated in the findings and in ``THRESHOLDS`` so they can be argued with.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np  # noqa: E402

G = 9.81
THRESHOLDS = dict(
    pos_xy_abs_m=60.0, z_min_m=0.0, z_max_m=10.0, speed_mps=20.0, omega_radps=10.0, rotor_speed_radps=(0.0, 838.0),
    tilt_deg=85.0, specific_force_ms2=4 * G,
    kin_pos_vel_mps_p99=0.10,          # |Δp/dt − (v_k+v_{k+1})/2|
    kin_acc_imu_ms2_p99=1.0,           # |Δv/dt − (R f + g)| (contact-free episodes); datasets with the raw Isaac IMU (v0.5.0)
    kin_acc_imu_corrected_ms2_p99=0.25,  # same, for datasets generated with physics.imu_transport_correction (v0.6.0 pilot: 0.051)
    kin_force_balance_ms2_p99=0.5,     # ground-truth Newton balance (independent of the IMU)
    kin_gyro_radps_p99=0.05,           # |IMU gyro − ω_B| (clean sensor)
    att_rate_rad_p99=0.02,             # |angle(q_k, q_{k+1}) − |ω| dt|
    corr_action_wind=0.3, corr_domain=0.5, ks_p=0.01, mix_tvd=0.25, min_episodes_per_cell=5,
)
STATE_NAMES = ["px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz"]


def finding(level: str, check: str, msg: str, **data) -> dict:
    return {"level": level, "check": check, "message": msg, **({"data": data} if data else {})}


def _f(x) -> float:
    return float(np.round(x, 6))


# ------------------------------------------------------------------ range
def check_range(episodes: Sequence, limits: Mapping = THRESHOLDS) -> tuple[list[dict], dict]:
    S = np.concatenate([e.states for e in episodes]); O = np.concatenate([e.obs for e in episodes])
    A = np.concatenate([e.actions for e in episodes])
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (S[:, 7] ** 2 + S[:, 8] ** 2), -1, 1)))
    speed = np.linalg.norm(S[:, 3:6], axis=1)
    chans = {n: S[:, i] for i, n in enumerate(STATE_NAMES)}
    chans.update({f"imu_f{a}": O[:, i] for i, a in enumerate("xyz")}); chans.update({f"imu_g{a}": O[:, 3 + i] for i, a in enumerate("xyz")})
    chans.update({f"rotor{i}": A[:, i] for i in range(4)}); chans["speed"], chans["tilt_deg"] = speed, tilt
    chans["specific_force_norm"] = np.linalg.norm(O[:, :3], axis=1); chans["omega_norm"] = np.linalg.norm(S[:, 10:13], axis=1)
    stats = {k: dict(min=_f(v.min()), p1=_f(np.percentile(v, 1)), p99=_f(np.percentile(v, 99)), max=_f(v.max()), std=_f(v.std()), n=int(len(v)))
             for k, v in chans.items()}
    out: list[dict] = []
    frozen = [k for k in STATE_NAMES + [f"rotor{i}" for i in range(4)] if stats[k]["std"] < 1e-9]
    out.append(finding("fail", "range.coverage", f"frozen channels (std = 0): {frozen}") if frozen
               else finding("pass", "range.coverage", "every state and rotor channel varies (std > 0)"))
    rules = [("pz", S[:, 2] < limits["z_min_m"], f"z below {limits['z_min_m']} m (ground penetration)"),
             ("pz", S[:, 2] > limits["z_max_m"], f"z above {limits['z_max_m']} m"),
             ("pxy", np.abs(S[:, 0:2]).max(1) > limits["pos_xy_abs_m"], f"|x| or |y| above {limits['pos_xy_abs_m']} m"),
             ("speed", speed > limits["speed_mps"], f"speed above {limits['speed_mps']} m/s"),
             ("omega", chans["omega_norm"] > limits["omega_radps"], f"‖ω‖ above {limits['omega_radps']} rad/s"),
             ("tilt", tilt > limits["tilt_deg"], f"tilt above {limits['tilt_deg']}°"),
             ("force", chans["specific_force_norm"] > limits["specific_force_ms2"], f"IMU specific force above {limits['specific_force_ms2']:.0f} m/s²"),
             ("rotor", (A < limits["rotor_speed_radps"][0]).any(1) | (A > limits["rotor_speed_radps"][1]).any(1), "rotor speed outside [0, Ω_max]")]
    for name, mask, text in rules:
        n = int(mask.sum())
        out.append(finding("warn", f"range.{name}", f"{n} samples ({100 * n / len(S):.3f} %): {text}") if n
                   else finding("pass", f"range.{name}", f"no sample: {text}"))
    finite = all(np.isfinite(x).all() for x in (S, O, A))
    out.append(finding("pass" if finite else "fail", "range.finite", "all values finite" if finite else "non-finite values present"))
    return out, stats


# ------------------------------------------------------------------ outlier episodes
def _episode_channels(e) -> dict[str, np.ndarray]:
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (e.states[:, 7] ** 2 + e.states[:, 8] ** 2), -1, 1)))
    return {"speed [m/s]": np.linalg.norm(e.states[:, 3:6], axis=1), "tilt [deg]": tilt, "‖ω‖ [rad/s]": np.linalg.norm(e.states[:, 10:13], axis=1),
            "IMU |f| [m/s²]": np.linalg.norm(e.obs[:, :3], axis=1), "rotor max [rad/s]": e.actions.max(1)}


def episode_outliers(episodes: Sequence, k_mad: float = 5.0, top: int = 3) -> dict:
    """Per channel, the per-episode p99 and the episodes whose p99 lies more than ``k_mad`` robust deviations (MAD·1.4826) above
    the median of all episodes. Detects single odd episodes that the pooled percentiles hide."""
    ids = [e.episode_id for e in episodes]
    per = {name: np.array([np.percentile(_episode_channels(e)[name], 99) for e in episodes]) for name in _episode_channels(episodes[0])}
    out = {}
    for name, v in per.items():
        med = float(np.median(v)); mad = float(np.median(np.abs(v - med)) * 1.4826) or 1e-12
        z = (v - med) / mad
        order = np.argsort(-z)
        out[name] = {"median_p99": med, "robust_sd": mad, "p99_by_episode": {i: float(x) for i, x in zip(ids, v)},
                     "outliers": [{"episode": ids[j], "p99": float(v[j]), "robust_z": float(z[j])} for j in order if z[j] > k_mad],
                     "top": [{"episode": ids[j], "p99": float(v[j]), "robust_z": float(z[j])} for j in order[:top]]}
    return out


def check_outliers(episodes: Sequence, k_mad: float = 5.0) -> tuple[list[dict], dict]:
    res = episode_outliers(episodes, k_mad)
    flagged = {n: [o["episode"] for o in v["outliers"]] for n, v in res.items() if v["outliers"]}
    if not flagged:
        return [finding("pass", "outliers.episodes", f"no episode has a per-channel p99 more than {k_mad:g} robust deviations above the median")], res
    # info, not warn: a robust-z outlier is a candidate to look at (e.g. an aggressive scenario), not an error by itself
    return [finding("info", "outliers.episodes", "episodes with an unusual per-episode p99 (> %g robust deviations above the median): %s" % (
        k_mad, "; ".join(f"{n}: {', '.join(v[:4])}{' …' if len(v) > 4 else ''} ({len(v)})" for n, v in flagged.items())))], res


# ------------------------------------------------------------------ duplicates
def _hash(arrays: Sequence[np.ndarray], decimals: int) -> str:
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.round(a, decimals).astype(np.float32).tobytes())
    return h.hexdigest()


def check_duplicates(episodes: Sequence, rows: Mapping[str, dict] | None = None) -> list[dict]:
    out: list[dict] = []
    exact, coarse = defaultdict(list), defaultdict(list)
    for e in episodes:
        exact[_hash([e.states[::5], e.actions[::5]], 5)].append(e.episode_id)
        coarse[_hash([e.states[::5, 0:3], e.actions[::5]], 1)].append(e.episode_id)
    d = [v for v in exact.values() if len(v) > 1]
    out.append(finding("fail", "duplicates.exact", f"{len(d)} groups of identical episodes: {d[:3]}") if d
               else finding("pass", "duplicates.exact", f"no identical episodes among {len(episodes)}"))
    near = []
    for v in coarse.values():
        if len(v) > 1:
            gids = {(rows or {}).get(i, {}).get("group_id", i) for i in v}
            if len(gids) > 1:                              # siblings of one counterfactual group may legitimately be close
                near.append(v)
    out.append(finding("warn", "duplicates.near", f"{len(near)} sets of near-identical episodes in different groups: {near[:3]}") if near
               else finding("pass", "duplicates.near", "no near-identical episodes across groups"))
    if rows:
        seeds = Counter(r["seeds"]["episode"] for r in rows.values())
        dup = [s for s, c in seeds.items() if c > 1]
        out.append(finding("fail", "duplicates.seeds", f"{len(dup)} repeated episode seeds") if dup else finding("pass", "duplicates.seeds", "episode seeds are unique"))
    return out


# ------------------------------------------------------------------ physical consistency
def _rot_from_q(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)


def physical_residuals(e) -> dict[str, np.ndarray]:
    dt = e.dt
    s, o = e.states, e.obs
    pv = np.linalg.norm((s[1:, 0:3] - s[:-1, 0:3]) / dt - 0.5 * (s[1:, 3:6] + s[:-1, 3:6]), axis=1)
    f_mid = 0.5 * (o[1:, :3] + o[:-1, :3])
    a_imu = np.einsum("nij,nj->ni", _rot_from_q(0.5 * (s[1:, 6:10] + s[:-1, 6:10]) / np.linalg.norm(0.5 * (s[1:, 6:10] + s[:-1, 6:10]), axis=1, keepdims=True)), f_mid) \
        + np.array([0, 0, -G])
    acc = np.linalg.norm((s[1:, 3:6] - s[:-1, 3:6]) / dt - a_imu, axis=1)
    gyro = np.linalg.norm(o[:, 3:6] - s[:, 10:13], axis=1)
    dq = np.abs((s[1:, 6:10] * s[:-1, 6:10]).sum(1)).clip(0, 1)
    ang = 2 * np.arccos(dq)
    att = np.abs(ang - 0.5 * (np.linalg.norm(s[1:, 10:13], axis=1) + np.linalg.norm(s[:-1, 10:13], axis=1)) * dt)
    return dict(pos_vel=pv, acc_imu=acc, gyro=gyro, att_rate=att,
                q_norm=np.abs(np.linalg.norm(s[:, 6:10], axis=1) - 1), q_sign=((s[1:, 6:10] * s[:-1, 6:10]).sum(1)))


def acc_imu_residual_raw(arrays: Mapping[str, np.ndarray], dt: float) -> np.ndarray:
    """|Δv/dt − (R(q̄)·f̄ + g)| at the physics rate (midpoint of each step); the model-rate version is too coarse for
    fast transients (rotor doublets, acceleration ramps) because the IMU is sampled only at the step ends."""
    v = arrays["ground_truth/velocity_W"].astype(np.float64); q = arrays["ground_truth/orientation_wxyz"].astype(np.float64)
    f = arrays["observations/imu_accel_B"].astype(np.float64)
    qm = 0.5 * (q[1:] + q[:-1]); qm /= np.linalg.norm(qm, axis=1, keepdims=True)
    a_imu = np.einsum("nij,nj->ni", _rot_from_q(qm), 0.5 * (f[1:] + f[:-1])) + np.array([0, 0, -G])
    return np.linalg.norm((v[1:] - v[:-1]) / dt - a_imu, axis=1)


def force_balance_residual_raw(arrays: Mapping[str, np.ndarray], mass: float, dt: float) -> np.ndarray:
    """Newton's law on the ground truth, independent of the IMU: |Δv/dt − (R·(ΣT/m)ẑ_B + g + F_ext/m)| per physics step
    (rotor thrust and attitude at the step midpoint, the injected external force of the interval)."""
    v = arrays["ground_truth/velocity_W"].astype(np.float64); q = arrays["ground_truth/orientation_wxyz"].astype(np.float64)
    T = arrays["ground_truth/rotor_thrust"].astype(np.float64).sum(1)
    Tm = 0.5 * (T[1:] + T[:-1]); qm = 0.5 * (q[1:] + q[:-1]); qm /= np.linalg.norm(qm, axis=1, keepdims=True)
    zb = _rot_from_q(qm)[:, :, 2]
    a = zb * (Tm / mass)[:, None] + np.array([0, 0, -G]) + arrays["ground_truth/wrench/external_force_W"].astype(np.float64) / mass
    return np.linalg.norm((v[1:] - v[:-1]) / dt - a, axis=1)


def check_physical(episodes: Sequence, statuses: Mapping[str, str] | None = None, limits: Mapping = THRESHOLDS,
                   acc_raw: Mapping[str, np.ndarray] | None = None,
                   balance_raw: Mapping[str, np.ndarray] | None = None, imu_corrected: bool = False) -> tuple[list[dict], dict]:
    """``imu_corrected``: the dataset was generated with the IMU transport correction, so the tight IMU limit applies."""
    limits = dict(limits)
    if imu_corrected:
        limits["kin_acc_imu_ms2_p99"] = limits["kin_acc_imu_corrected_ms2_p99"]
    acc: dict[str, list[np.ndarray]] = defaultdict(list)
    for e in episodes:
        r = physical_residuals(e)
        for k, v in r.items():
            if k == "acc_imu":
                if (statuses or {}).get(e.episode_id, "completed") != "completed":
                    continue                               # contact episodes: the IMU sees impact forces
                if acc_raw is not None and e.episode_id in acc_raw:
                    v = acc_raw[e.episode_id]              # physics-rate residual
            acc[k].append(v)
    cat = {k: np.concatenate(v) for k, v in acc.items()}
    summ = {k: dict(p50=_f(np.percentile(v, 50)), p99=_f(np.percentile(v, 99)), max=_f(v.max())) for k, v in cat.items() if k != "q_sign"}
    out = []
    for key, thr_key, text, unit in (("pos_vel", "kin_pos_vel_mps_p99", "Δp/dt vs mean velocity", "m/s"),
                                     ("acc_imu", "kin_acc_imu_ms2_p99", "Δv/dt vs R·(IMU specific force) + g at the physics rate" if acc_raw is not None else "Δv/dt vs R·(IMU specific force) + g at the model rate", "m/s²"),
                                     ("gyro", "kin_gyro_radps_p99", "IMU gyro vs true ω_B", "rad/s"),
                                     ("att_rate", "att_rate_rad_p99", "rotation angle per step vs ‖ω‖·dt", "rad")):
        v = summ[key]["p99"]
        out.append(finding("pass" if v <= limits[thr_key] else "warn", f"physical.{key}",
                           f"{text}: p50 {summ[key]['p50']:.4g}, p99 {v:.4g}, max {summ[key]['max']:.4g} {unit} (limit p99 {limits[thr_key]})"
                           + (" — the Isaac IMU reports specific force − ω_B×v_B (R² 0.83 on the Run 5 pilot; the tail tracks ‖ω‖ and the ground-truth force balance holds); "
                              "datasets from v0.6.0 on are generated with physics.imu_transport_correction=true" if key == "acc_imu" and v > limits[thr_key] and not imu_corrected else "")
                           + (" [IMU transport correction applied at generation: tight limit]" if key == "acc_imu" and imu_corrected else "")))
    if balance_raw:
        b = np.concatenate([v for k, v in balance_raw.items() if (statuses or {}).get(k, "completed") == "completed"])
        summ["force_balance"] = dict(p50=_f(np.percentile(b, 50)), p99=_f(np.percentile(b, 99)), max=_f(b.max()))
        out.append(finding("pass" if summ["force_balance"]["p99"] <= limits["kin_force_balance_ms2_p99"] else "fail", "physical.force_balance",
                           f"Newton's law on the ground truth, Δv/dt vs R·(ΣT/m)ẑ + g + F_ext/m: p50 {summ['force_balance']['p50']:.4g}, "
                           f"p99 {summ['force_balance']['p99']:.4g}, max {summ['force_balance']['max']:.4g} m/s² (limit p99 {limits['kin_force_balance_ms2_p99']})"))
    out.append(finding("pass" if summ["q_norm"]["max"] < 1e-4 else "fail", "physical.quaternion_norm", f"max | ‖q‖ − 1 | = {summ['q_norm']['max']:.2e}"))
    flips = int((cat["q_sign"] < 0).sum())
    out.append(finding("pass" if flips == 0 else "fail", "physical.quaternion_sign", f"{flips} sign discontinuities between consecutive quaternions"))
    return out, summ


# ------------------------------------------------------------------ correlation / identifiability
def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _pooled_action_wind(eps_u: Sequence, eps_w: Sequence) -> float:
    n = [min(len(u.actions), len(w.aux["environment/wind_applied_W"])) for u, w in zip(eps_u, eps_w)]
    U = np.concatenate([u.actions[:k] for u, k in zip(eps_u, n)]); W = np.concatenate([w.aux["environment/wind_applied_W"][:k] for w, k in zip(eps_w, n)])
    return float(np.abs(np.array([[corr(U[:, i], W[:, j]) for j in range(3)] for i in range(4)])).max())


def action_wind_permutation_p(eps: Sequence, n_perm: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(observed max |ρ|, null 95th percentile, p-value) with the wind series shuffled between episodes: how often chance alone
    gives a pooled correlation at least as large with this few episodes. A fixed limit ignores the sample size."""
    eps = sorted(eps, key=lambda e: e.episode_id)          # independent of how the shards list the episodes (parallel workers reorder them)
    obs = _pooled_action_wind(eps, eps)
    rng = np.random.default_rng(seed)
    null = np.array([_pooled_action_wind(eps, [eps[i] for i in rng.permutation(len(eps))]) for _ in range(n_perm)])
    return obs, float(np.percentile(null, 95)), float((null >= obs).mean())


def check_correlation(episodes: Sequence, rows: Mapping[str, dict], physics=None, limits: Mapping = THRESHOLDS) -> tuple[list[dict], dict]:
    out: list[dict] = []
    res: dict = {"action_wind": {}, "domain": {}, "features": {}, "residual_vs_wind": {}}
    by_type = defaultdict(list)
    for e in episodes:
        if "environment/wind_applied_W" in e.aux:
            by_type[rows[e.episode_id]["data_type"]].append(e)
    for dtype, eps in by_type.items():
        U = np.concatenate([e.actions[: len(e.aux["environment/wind_applied_W"])] for e in eps])
        W = np.concatenate([e.aux["environment/wind_applied_W"][: len(e.actions)] for e in eps])
        n = min(len(U), len(W)); U, W = U[:n], W[:n]
        Ud = np.concatenate([e.actions - e.actions.mean(0) for e in eps])[:n]; Wd = np.concatenate([e.aux["environment/wind_applied_W"][: len(e.actions)] - e.aux["environment/wind_applied_W"][: len(e.actions)].mean(0) for e in eps])[:n]
        C = np.array([[corr(U[:, i], W[:, j]) for j in range(3)] for i in range(4)])
        Cd = np.array([[corr(Ud[:, i], Wd[:, j]) for j in range(3)] for i in range(4)])
        thrust = (U ** 2).sum(1)
        c_thr = corr(thrust, np.linalg.norm(W, axis=1))
        res["action_wind"][dtype] = dict(matrix=C.round(3).tolist(), max_abs=_f(np.abs(C).max()), within_episode_max_abs=_f(np.abs(Cd).max()),
                                         thrust_vs_speed=_f(c_thr), n_episodes=len(eps), n_samples=int(n))
        worst = max(np.abs(C).max(), abs(c_thr))
        perm = ""
        if len(eps) >= 6:
            o_, p95_, p_ = action_wind_permutation_p(eps)
            res["action_wind"][dtype].update(permutation_null_p95=p95_, permutation_p=p_)
            perm = f"; with the wind shuffled between episodes chance gives up to {p95_:.2f} (95th percentile), permutation p = {p_:.3f}"
        out.append(finding("pass" if worst <= limits["corr_action_wind"] else "warn", f"correlation.action_wind.{dtype}",
                           f"max |ρ(rotor speed, wind)| = {np.abs(C).max():.2f}, within-episode {np.abs(Cd).max():.2f}, ρ(thrust, |wind|) = {c_thr:.2f} "
                           f"over {len(eps)} episodes (limit {limits['corr_action_wind']}; the action must not encode the wind){perm}"))
    # domain factors across distinct ID domains (the OOD shells are sampled differently and are not mixed in)
    doms: dict[str, dict] = {}
    for r in rows.values():
        if r["domain"].get("region") == "id":
            doms.setdefault(r["domain"]["domain_id"], r["domain"].get("factors") or {})
    names = sorted({k for d in doms.values() for k in d})
    if len(doms) >= 8 and names:
        M = np.array([[d.get(k, 1.0) for k in names] for d in doms.values()])
        C = np.nan_to_num(np.corrcoef(M.T)); np.fill_diagonal(C, 0)
        n = len(doms); i, j = np.unravel_index(np.abs(C).argmax(), C.shape); r_max = abs(C[i, j])
        from math import atanh, erf, sqrt
        z = atanh(min(r_max, 0.999999)) * sqrt(max(n - 3, 1))
        p_one = 1 - 0.5 * (1 + erf(z / sqrt(2))); n_pairs = len(names) * (len(names) - 1) // 2
        p_bonf = min(1.0, 2 * p_one * n_pairs)
        res["domain"] = dict(names=names, matrix=C.round(3).tolist(), max_abs=_f(r_max), pair=[names[i], names[j]], n_domains=n, bonferroni_p=_f(p_bonf))
        significant = p_bonf < 0.05 and r_max > limits["corr_domain"]
        out.append(finding("warn" if significant else "pass", "correlation.domain_factors",
                           f"largest |ρ| between ID domain factors = {r_max:.2f} ({names[i]} × {names[j]}) over {n} distinct ID domains; Bonferroni p = {p_bonf:.3f} "
                           f"({'above the limit and significant: sample more domains or decorrelate the design' if significant else 'consistent with chance or below the limit ' + str(limits['corr_domain'])})"))
    F = np.concatenate([np.concatenate([e.states[:-1, 3:6], e.states[:-1, 10:13], e.obs[:-1], e.actions], axis=1) for e in episodes])
    fn = ["vx", "vy", "vz", "wx", "wy", "wz", "fx", "fy", "fz", "gx", "gy", "gz", "r0", "r1", "r2", "r3"]
    C = np.nan_to_num(np.corrcoef(F.T)); np.fill_diagonal(C, 0)
    pairs = [(fn[i], fn[j], _f(C[i, j])) for i in range(len(fn)) for j in range(i + 1, len(fn)) if abs(C[i, j]) > 0.9]
    res["features"] = dict(names=fn, matrix=C.round(3).tolist(), highly_correlated=pairs)
    out.append(finding("info", "correlation.features", f"{len(pairs)} feature pairs with |ρ| > 0.9: {pairs[:6]}"))
    # does the residual of nominal physics carry the wind signal?  (identifiability: Δ = s_{t+1} − f_phys must grow with wind)
    if physics is not None:
        import torch
        from physiworld_uav.models.physics import rk4_step
        for dtype, eps in by_type.items():
            dv, ws = [], []
            for e in eps:
                s, u, t = torch.tensor(e.states[:-1]), torch.tensor(e.actions), torch.tensor(e.states[1:])
                o = rk4_step(lambda x, u=u: physics.derivative(x, u), s, e.dt, physics.project)
                d = ((t - o)[:, 3:6] / e.dt).norm(dim=1).numpy(); w = np.linalg.norm(e.aux["environment/wind_applied_W"], axis=1)
                n = min(len(d), len(w)); dv.append(d[:n]); ws.append(w[:n])
            dv, ws = np.concatenate(dv), np.concatenate(ws)
            edges = [0, 0.5, 1, 2, 3, 4, 6, 8.5]
            bins = [dict(lo=a, hi=b, n=int(((ws >= a) & (ws < b)).sum()), mean_residual_ms2=_f(dv[(ws >= a) & (ws < b)].mean()))
                    for a, b in zip(edges[:-1], edges[1:]) if ((ws >= a) & (ws < b)).sum() > 50]
            r = corr(dv, ws)
            res["residual_vs_wind"][dtype] = dict(corr=_f(r), bins=bins)
            out.append(finding("pass" if r > 0.2 else "warn", f"identifiability.residual_vs_wind.{dtype}",
                               f"ρ(‖Δv‖, |wind|) = {r:.2f}; mean ‖Δv‖ rises from {bins[0]['mean_residual_ms2']:.2f} m/s² (calm) to "
                               f"{bins[-1]['mean_residual_ms2']:.2f} m/s² (strongest wind); the calm-air floor is parameter mismatch and unmodelled drag"))
    return out, res


# ------------------------------------------------------------------ distribution (intended vs realised)
def _ks_uniform(x: np.ndarray, lo: float, hi: float) -> float:
    try:
        from scipy.stats import kstest
        return float(kstest((x - lo) / (hi - lo), "uniform").pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def check_distribution(rows: Mapping[str, dict], domain_cfg: Mapping | None, experiment_cfg: Mapping | None, limits: Mapping = THRESHOLDS) -> tuple[list[dict], dict]:
    out: list[dict] = []; res: dict = {"wind": {}, "domain": {}, "mix": {}}
    # wind: realised statistics inside their regime limits
    by_reg = defaultdict(list)
    for r in rows.values():
        w = r.get("wind") or {}
        if w.get("stats"):
            by_reg[w.get("regime", "none")].append(w)
    for reg, ws in by_reg.items():
        mean_h = np.array([w["stats"]["mean_horizontal_speed"] for w in ws]); peak = np.array([w["stats"]["peak_speed"] for w in ws])
        lim = (ws[0].get("limits") or {})
        mh = lim.get("mean_horizontal"); pk = lim.get("peak_speed_max")
        ok = bool(((mean_h >= mh[0] - 1e-9) & (mean_h <= mh[1] + 1e-9)).all()) if mh else True
        ok_pk = bool((peak <= pk + 1e-9).all()) if pk is not None else True
        res["wind"][reg] = dict(n=len(ws), mean_horizontal=dict(min=_f(mean_h.min()), median=_f(np.median(mean_h)), max=_f(mean_h.max())),
                                peak=dict(max=_f(peak.max())), limits=lim)
        out.append(finding("pass" if ok and ok_pk else "fail", f"distribution.wind.{reg}",
                           f"{len(ws)} episodes: mean horizontal speed {mean_h.min():.2f}–{mean_h.max():.2f} m/s (limit {mh}), peak ≤ {peak.max():.2f} m/s (limit {pk})"))
    # domain factors against the intended ranges
    if domain_cfg:
        params = domain_cfg.get("parameters", {})
        per_region = defaultdict(dict)
        for r in rows.values():
            d = r["domain"]; per_region[d.get("region", "?")][d["domain_id"]] = d.get("factors") or {}
        for region, doms in per_region.items():
            for name, spec in params.items():
                vals = np.array([d[name] for d in doms.values() if name in d])
                if not len(vals):
                    continue
                hw = spec.get("id_halfwidth", 0.0); shell = spec.get("ood_shell")
                delta = vals - 1.0
                if region in ("id", "nominal"):
                    inside = float(np.mean(np.abs(delta) <= hw + 1e-9)) if region == "id" else float(np.mean(np.abs(delta) < 1e-9))
                    p = _ks_uniform(vals, 1 - hw, 1 + hw) if region == "id" and len(vals) >= 5 and hw > 0 else float("nan")
                elif region == "ood_physics" and shell:
                    inside = float(np.mean((np.abs(delta) > shell[0] - 1e-9) & (np.abs(delta) <= shell[1] + 1e-9)))
                    p = float("nan")
                else:
                    inside, p = float("nan"), float("nan")
                res["domain"].setdefault(region, {})[name] = dict(n=len(vals), min=_f(vals.min()), max=_f(vals.max()), fraction_in_intended_range=_f(inside), ks_uniform_p=_f(p))
        bad = [(g, k, v["fraction_in_intended_range"]) for g, d in res["domain"].items() for k, v in d.items()
               if v["fraction_in_intended_range"] == v["fraction_in_intended_range"] and v["fraction_in_intended_range"] < 1.0 and g in ("id", "nominal")]
        out.append(finding("fail" if bad else "pass", "distribution.domain_ranges", f"ID domain factors outside ±halfwidth: {bad[:5]}" if bad else
                           "every ID domain factor is inside its intended range"))
        low_p = [(k, v["n"], v["ks_uniform_p"]) for k, v in res["domain"].get("id", {}).items() if v["ks_uniform_p"] == v["ks_uniform_p"] and v["ks_uniform_p"] < limits["ks_p"]]
        out.append(finding("warn" if low_p else "pass", "distribution.domain_uniform",
                           f"KS test against uniform rejects (p < {limits['ks_p']}): {low_p}" if low_p else
                           "no ID domain factor rejects uniformity (KS, small n: weak evidence either way)"))
    # scenario mix vs planned weights
    if experiment_cfg:
        for part in experiment_cfg.get("parts", []):
            want = {m["scenario"]: float(m["weight"]) for m in part.get("mix", [])}
            if not want:
                continue
            tot = sum(want.values()); want = {k: v / tot for k, v in want.items()}
            got = Counter(r["scenario_template"] for r in rows.values() if r.get("part") == part["name"] and r.get("replay_of") is None)
            n = sum(got.values())
            if not n:
                continue
            tvd = 0.5 * sum(abs(got.get(k, 0) / n - want.get(k, 0)) for k in set(want) | set(got))
            res["mix"][part["name"]] = dict(n=n, tvd=_f(tvd), realised={k: v for k, v in sorted(got.items())}, planned={k: _f(v) for k, v in want.items()})
            out.append(finding("pass" if tvd <= limits["mix_tvd"] else "warn", f"distribution.mix.{part['name']}",
                               f"scenario mix vs planned weights: total-variation distance {tvd:.2f} over {n} baseline episodes (limit {limits['mix_tvd']})"))
    return out, res


# ------------------------------------------------------------------ coverage
def _occupancy(x: np.ndarray, lo: float, hi: float, bins: int) -> tuple[float, list[int]]:
    h, _ = np.histogram(x, bins=bins, range=(lo, hi))
    return float((h > 0).mean()), h.astype(int).tolist()


def check_coverage(rows: Mapping[str, dict], episodes: Sequence, domain_cfg: Mapping | None, limits: Mapping = THRESHOLDS) -> tuple[list[dict], dict]:
    out: list[dict] = []; res: dict = {}
    # (1) vehicle-domain factors, per factor over the ID range (10 bins) and 2-D pairs (5x5)
    doms = {}
    for r in rows.values():
        if r["domain"].get("region") == "id":
            doms[r["domain"]["domain_id"]] = r["domain"].get("factors") or {}
    if domain_cfg and doms:
        params = domain_cfg.get("parameters", {}); res["domain_1d"] = {}
        for name, spec in params.items():
            hw = spec.get("id_halfwidth")
            vals = np.array([d[name] for d in doms.values() if name in d])
            if hw and len(vals):
                occ, h = _occupancy(vals, 1 - hw, 1 + hw, 10)
                res["domain_1d"][name] = dict(occupied_fraction=_f(occ), counts=h, n=len(vals))
        pairs = [("mass", "force_scale"), ("mass", "motor_time_constant"), ("force_scale", "rotor_drag_scale"), ("mass", "body_drag_scale")]
        res["domain_2d"] = {}
        for a, b in pairs:
            if a in params and b in params and params[a].get("id_halfwidth") and params[b].get("id_halfwidth"):
                xa = np.array([d[a] for d in doms.values()]); xb = np.array([d[b] for d in doms.values()])
                h, _, _ = np.histogram2d(xa, xb, bins=5, range=[(1 - params[a]["id_halfwidth"], 1 + params[a]["id_halfwidth"]), (1 - params[b]["id_halfwidth"], 1 + params[b]["id_halfwidth"])])
                res["domain_2d"][f"{a}×{b}"] = dict(occupied_fraction=_f((h > 0).mean()), n_domains=len(xa))
        worst = min(v["occupied_fraction"] for v in res["domain_1d"].values())
        two = min(v["occupied_fraction"] for v in res["domain_2d"].values()) if res["domain_2d"] else 1.0
        out.append(finding("pass" if worst >= 0.7 else "warn", "coverage.domain_1d",
                           f"{len(doms)} distinct ID domains: 10-bin occupancy per factor ≥ {worst:.0%} (target ≥ 70 %)"))
        out.append(finding("pass" if two >= 0.5 else "warn", "coverage.domain_2d",
                           f"2-D (5×5) occupancy of factor pairs ≥ {two:.0%} (target ≥ 50 %; sparse by construction when n_domains is small)"))
    # (2) wind
    ws = [r["wind"] for r in rows.values() if (r.get("wind") or {}).get("stats")]
    if ws:
        speed = np.array([w["stats"]["mean_horizontal_speed"] for w in ws])
        az = np.array([np.degrees(np.arctan2(w["stats"]["mean_y"], w["stats"]["mean_x"])) % 360 for w in ws])
        el = np.array([w["stats"]["mean_elevation_deg"] for w in ws])
        res["wind"] = dict(speed_occupancy=_occupancy(speed, 0, 8.5, 17), azimuth_occupancy=_occupancy(az[speed > 0.3], 0, 360, 12),
                           elevation_occupancy=_occupancy(el[speed > 0.3], -60, 60, 8),
                           profiles=dict(Counter(w.get("profile", "none") for w in ws)),
                           gust_class=dict(Counter((w.get("labels") or {}).get("gust_class", "none") for w in ws)),
                           temporal_class=dict(Counter((w.get("labels") or {}).get("temporal_class", "none") for w in ws)))
        s_occ, a_occ, e_occ = res["wind"]["speed_occupancy"][0], res["wind"]["azimuth_occupancy"][0], res["wind"]["elevation_occupancy"][0]
        out.append(finding("pass" if s_occ >= 0.7 else "warn", "coverage.wind_speed", f"{s_occ:.0%} of 17 half-m/s bins in 0–8.5 m/s contain an episode"))
        out.append(finding("pass" if a_occ >= 0.8 else "warn", "coverage.wind_azimuth", f"{a_occ:.0%} of 12 azimuth sectors (windy episodes) are covered"))
        out.append(finding("info", "coverage.wind_elevation", f"{e_occ:.0%} of 8 elevation bins in ±60° are covered (ID wind is near-horizontal by design; vertical wind is OOD)"))
    # (3) scenario × role cells
    cells = Counter((r.get("scenario_template", "?"), r["split_role"]) for r in rows.values() if r.get("replay_of") is None)
    res["scenario_role"] = {f"{s}|{r}": n for (s, r), n in sorted(cells.items())}
    per_role = defaultdict(set); per_role_n = Counter()
    for r in rows.values():
        per_role[r["split_role"]].add(r["group_id"]); per_role_n[r["split_role"]] += 1
    res["groups_per_role"] = {k: len(v) for k, v in sorted(per_role.items())}
    few = {k: len(v) for k, v in per_role.items() if len(v) < limits["min_episodes_per_cell"]}
    out.append(finding("warn" if few else "pass", "coverage.groups_per_role",
                       f"roles with fewer than {limits['min_episodes_per_cell']} independent groups (too few to report per-role results with confidence): {few}" if few else
                       "every role has enough independent groups"))
    # (4) state-space coverage and OOD novelty relative to the training pool
    role_of = {e.episode_id: rows[e.episode_id]["split_role"] for e in episodes}
    feats = {}
    for e in episodes:
        s = e.states
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (s[:, 7] ** 2 + s[:, 8] ** 2), -1, 1)))
        f = np.stack([np.linalg.norm(s[:, 3:6], axis=1), tilt, np.linalg.norm(s[:, 10:13], axis=1), s[:, 2], e.obs[:, 2], np.r_[e.actions[:, 0], e.actions[-1, 0]]], 1)
        feats.setdefault(role_of[e.episode_id], []).append(f)
    names = ["speed", "tilt_deg", "omega_norm", "z", "imu_fz", "rotor0"]
    train = np.concatenate(feats["train"]) if "train" in feats else None
    if train is not None:
        lo, hi = np.percentile(train, 1, axis=0), np.percentile(train, 99, axis=0)
        res["novelty_vs_train"] = {}
        for role, fl in sorted(feats.items()):
            if role == "train":
                continue
            F = np.concatenate(fl)
            out_frac = ((F < lo) | (F > hi)).mean(0)
            res["novelty_vs_train"][role] = {n: _f(v) for n, v in zip(names, out_frac)}
        mag = res["novelty_vs_train"].get("test_ood_magnitude")
        idr = res["novelty_vs_train"].get("test_id")
        if mag and idr:
            out.append(finding("info", "coverage.novelty",
                               "fraction of test samples outside the training 1–99 % range (speed / tilt / ‖ω‖ / z / IMU f_z / rotor 0): "
                               f"test_id {[idr[n] for n in names]}; test_ood_magnitude {[mag[n] for n in names]}"))
    res["state_space_names"] = names
    # (5) initial conditions
    p0 = np.array([r["initial_conditions"]["position"] for r in rows.values() if r.get("initial_conditions")])
    yaw = np.array([r["initial_conditions"]["yaw_deg"] for r in rows.values() if r.get("initial_conditions")]) % 360
    res["initial"] = dict(z=dict(min=_f(p0[:, 2].min()), max=_f(p0[:, 2].max())), yaw_occupancy=_occupancy(yaw, 0, 360, 8)[0])
    out.append(finding("info", "coverage.initial_conditions", f"initial altitude {p0[:, 2].min():.2f}–{p0[:, 2].max():.2f} m, 8-sector yaw occupancy {res['initial']['yaw_occupancy']:.0%}"))
    # (6) effective sample size
    res["effective"] = {"episodes": len(rows), "groups": len({r["group_id"] for r in rows.values()}), "domains": len({r["domain"]["domain_id"] for r in rows.values()}),
                        "scenarios": len({r["scenario_id"] for r in rows.values()})}
    out.append(finding("info", "coverage.effective_size", f"{res['effective']['episodes']} episodes = {res['effective']['groups']} independent groups, "
                       f"{res['effective']['domains']} domains, {res['effective']['scenarios']} scenario instances; windows within an episode overlap and are not independent"))
    return out, res


# ------------------------------------------------------------------ driver and report
def run_all(dataset_dir: str | Path, with_physics: bool = True) -> dict:
    import yaml
    from physiworld_uav.data.dataset_builder import BuilderConfig, dataset_decimation, load_model_episodes
    dataset_dir = Path(dataset_dir)
    rows = {json.loads(l)["episode_id"]: json.loads(l) for l in (dataset_dir / "manifest.jsonl").read_text().splitlines() if l.strip()}
    index = json.loads((dataset_dir / "index.json").read_text())
    status = {e["episode_id"]: e.get("status", "?") for e in index["episodes"]}
    episodes = load_model_episodes(dataset_dir, BuilderConfig(decimation=dataset_decimation(dataset_dir)))
    rows_w = {k: v for k, v in rows.items() if k in status}
    cfg_path = dataset_dir / "resolved_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    domain_cfg = cfg.get("domain") if isinstance(cfg.get("domain"), dict) and "parameters" in cfg.get("domain", {}) else None
    if domain_cfg is None:
        dp = Path(__file__).resolve().parents[3] / "config" / "domain" / "randomized.yaml"
        domain_cfg = yaml.safe_load(dp.read_text()) if dp.exists() else None
    physics = None
    if with_physics:
        from physiworld_uav.models.physics import QuadrotorPhysics
        physics = QuadrotorPhysics.from_yaml()
    findings: list[dict] = []; metrics: dict = {}
    from physiworld_uav.data.writer import read_episode
    acc_raw, balance_raw, imu_flags = {}, {}, []
    for row in index["episodes"]:
        if row.get("status") == "completed":
            arrays, meta = read_episode(dataset_dir / row["shard"], row["episode_id"])
            imu_flags.append(bool((meta.sim or {}).get("imu_transport_correction", False)))
            dt_p = float(meta.sim.get("physics_dt", 0.005))
            acc_raw[row["episode_id"]] = acc_imu_residual_raw(arrays, dt_p)
            balance_raw[row["episode_id"]] = force_balance_residual_raw(arrays, float(meta.drone["mass_kg"]), dt_p)
    for name, fn in (("range", lambda: check_range(episodes)), ("duplicates", lambda: (check_duplicates(episodes, rows_w), {})), ("outliers", lambda: check_outliers(episodes)),
                     ("physical", lambda: check_physical(episodes, status, acc_raw=acc_raw, balance_raw=balance_raw,
                                                         imu_corrected=bool(imu_flags) and all(imu_flags))), ("correlation", lambda: check_correlation(episodes, rows_w, physics)),
                     ("distribution", lambda: check_distribution(rows_w, domain_cfg, cfg.get("experiment") or cfg)),
                     ("coverage", lambda: check_coverage(rows_w, episodes, domain_cfg))):
        f, m = fn(); findings += f; metrics[name] = m
    counts = Counter(f["level"] for f in findings)
    return {"dataset": str(dataset_dir), "n_manifest": len(rows), "n_written": len(status), "n_model_ready": len(episodes),
            "excluded_from_model_view": {k: v for k, v in status.items() if v != "completed"}, "summary": dict(counts),
            "findings": findings, "metrics": metrics, "thresholds": {k: v for k, v in THRESHOLDS.items()}}


def to_markdown(r: dict) -> str:
    s = r["summary"]
    lines = [f"# Data quality report — `{Path(r['dataset']).name}`", "",
             f"{r['n_written']} episodes written ({r['n_model_ready']} model-ready, {len(r['excluded_from_model_view'])} excluded: ground contact). "
             f"**{s.get('fail', 0)} fail · {s.get('warn', 0)} warn · {s.get('pass', 0)} pass · {s.get('info', 0)} info**", ""]
    for lvl, title in (("fail", "Failures"), ("warn", "Warnings"), ("info", "Information"), ("pass", "Passed")):
        items = [f for f in r["findings"] if f["level"] == lvl]
        if items:
            lines += [f"## {title} ({len(items)})", ""] + [f"* `{f['check']}` — {f['message']}" for f in items] + [""]
    lines += ["Figures: see the PNG files next to this report. Thresholds are in `quality.json` (`thresholds`)."]
    return "\n".join(lines)


def make_figures(r: dict, dataset_dir: Path, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from physiworld_uav.data.dataset_builder import BuilderConfig, dataset_decimation, load_model_episodes
    rows = {json.loads(l)["episode_id"]: json.loads(l) for l in (dataset_dir / "manifest.jsonl").read_text().splitlines() if l.strip()}
    eps = load_model_episodes(dataset_dir, BuilderConfig(decimation=dataset_decimation(dataset_dir)))
    fam = lambda role: "train" if role == "train" else ("id (val/test)" if role in ("val_id", "test_id") else ("physics_validation" if role == "physics_validation" else "ood"))  # noqa: E731
    groups = defaultdict(list)
    for e in eps:
        groups[fam(rows[e.episode_id]["split_role"])].append(e)
    files = []
    # 1 histograms of state / IMU / action channels by role family
    chans = [("speed", lambda e: np.linalg.norm(e.states[:, 3:6], axis=1)), ("z [m]", lambda e: e.states[:, 2]),
             ("tilt [deg]", lambda e: np.degrees(np.arccos(np.clip(1 - 2 * (e.states[:, 7] ** 2 + e.states[:, 8] ** 2), -1, 1)))),
             ("‖ω‖ [rad/s]", lambda e: np.linalg.norm(e.states[:, 10:13], axis=1)), ("IMU f_z [m/s²]", lambda e: e.obs[:, 2]),
             ("rotor 0 [rad/s]", lambda e: e.actions[:, 0])]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for a, (name, fn) in zip(ax.flat, chans):
        for g, el in sorted(groups.items()):
            a.hist(np.concatenate([fn(e) for e in el]), bins=50, histtype="step", density=True, label=g)
        a.set(title=name, yscale="log"); a.grid(alpha=.25)
    ax[0, 0].legend(fontsize=8); fig.suptitle("Distributions by split family (density, log scale)"); fig.savefig(out / "hist_channels.png", dpi=120); plt.close(fig); files.append("hist_channels.png")
    # 2 correlations
    cm = r["metrics"]["correlation"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for dt, v in cm.get("action_wind", {}).items():
        pass
    aw = cm.get("action_wind", {})
    for a, dt in zip(ax[:2], ["closed_loop", "open_loop"]):
        if dt in aw:
            im = a.imshow(np.array(aw[dt]["matrix"]), vmin=-1, vmax=1, cmap="RdBu_r"); a.set(title=f"ρ(rotor i, wind axis) — {dt}", xticks=[0, 1, 2], xticklabels=["wx", "wy", "wz"], yticks=range(4), yticklabels=["r0", "r1", "r2", "r3"])
    if cm.get("domain"):
        im = ax[2].imshow(np.array(cm["domain"]["matrix"]), vmin=-1, vmax=1, cmap="RdBu_r"); ax[2].set(title="ρ between domain factors"); ax[2].set_xticks(range(len(cm["domain"]["names"]))); ax[2].set_xticklabels(cm["domain"]["names"], rotation=90, fontsize=7); ax[2].set_yticks(range(len(cm["domain"]["names"]))); ax[2].set_yticklabels(cm["domain"]["names"], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=.8); fig.savefig(out / "correlations.png", dpi=120); plt.close(fig); files.append("correlations.png")
    # 3 residual vs wind
    fig, a = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for dt, v in cm.get("residual_vs_wind", {}).items():
        a.plot([(b["lo"] + b["hi"]) / 2 for b in v["bins"]], [b["mean_residual_ms2"] for b in v["bins"]], "o-", label=f"{dt} (ρ={v['corr']:.2f})")
    a.set(xlabel="|wind| [m/s]", ylabel="mean ‖Δv_residual‖ [m/s²]", title="Nominal-physics residual vs wind (identifiability)"); a.legend(); a.grid(alpha=.25)
    fig.savefig(out / "residual_vs_wind.png", dpi=120); plt.close(fig); files.append("residual_vs_wind.png")
    # 4 coverage: wind, domain factors, scenario × role
    cov = r["metrics"]["coverage"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    if "wind" in cov:
        ax[0].bar(np.arange(17) * 0.5 + .25, cov["wind"]["speed_occupancy"][1], width=.45); ax[0].set(title="episodes per mean-wind bin", xlabel="m/s")
    d1 = cov.get("domain_1d", {})
    if d1:
        M1 = np.array([v["counts"] for v in d1.values()]); im1 = ax[1].imshow(M1, cmap="Blues", aspect="auto")
        ax[1].set_yticks(range(len(d1))); ax[1].set_yticklabels(list(d1), fontsize=7); ax[1].set_xlabel("bin across the ID range (low → high)")
        ax[1].set_title("ID domains per factor bin"); fig.colorbar(im1, ax=ax[1], shrink=.8)
    cells = cov["scenario_role"]; scn = sorted({k.split("|")[0] for k in cells}); roles = sorted({k.split("|")[1] for k in cells})
    M = np.array([[cells.get(f"{s}|{ro}", 0) for ro in roles] for s in scn]); im = ax[2].imshow(M, cmap="Blues")
    ax[2].set(xticks=range(len(roles)), yticks=range(len(scn))); ax[2].set_xticklabels(roles, rotation=90, fontsize=7); ax[2].set_yticklabels(scn, fontsize=7); ax[2].set_title("baseline episodes: scenario × role")
    fig.colorbar(im, ax=ax[2], shrink=.8); fig.savefig(out / "coverage.png", dpi=120); plt.close(fig); files.append("coverage.png")
    # 5 physical consistency
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)
    pr = defaultdict(list)
    for e in eps:
        for k, v in physical_residuals(e).items():
            if k in ("pos_vel", "acc_imu", "gyro", "att_rate"):
                pr[k].append(v)
    for a, (k, t) in zip(ax, [("pos_vel", "Δp/dt − v̄ [m/s]"), ("acc_imu", "Δv/dt − (R f + g) [m/s²]"), ("gyro", "IMU gyro − ω_B [rad/s]"), ("att_rate", "angle − ‖ω‖dt [rad]")]):
        a.hist(np.concatenate(pr[k]), bins=80, log=True); a.set(title=t)
    fig.savefig(out / "physical_consistency.png", dpi=120); plt.close(fig); files.append("physical_consistency.png")
    return files


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(prog="statistical-validator", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    out = a.out or Path("outputs/reports") / f"{a.dataset.name}_quality"
    out.mkdir(parents=True, exist_ok=True)
    r = run_all(a.dataset)
    (out / "quality.json").write_text(json.dumps(r, indent=2, default=float))
    r["figures"] = [] if a.no_figures else make_figures(r, a.dataset, out)
    (out / "quality_report.md").write_text(to_markdown(r))
    print(to_markdown(r)); print(f"\nwrote {out}")
    return r


if __name__ == "__main__":
    main()
