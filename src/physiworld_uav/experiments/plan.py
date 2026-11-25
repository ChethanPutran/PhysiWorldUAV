"""Experiment planning: config → manifest of fully specified episodes (plan §4, §6.5-DS8, §6.6).

Everything random is decided here, before any simulation, from the seed tree:
domains, initial conditions, references, excitation, wind realisations and
counterfactual siblings. The simulator run is then a pure function of a
manifest row, which makes episodes individually reproducible and lets
parallel workers run any subset in any order.

Hierarchy (design doc §13)::

    dataset ── part (one split role or a role mix, one wind regime, one domain spec)
                └ group (scenario instance; counterfactual group or single episode)
                    └ episode (one simulation; sibling of the group)

Split roles are assigned to whole **groups**, and randomised domains are
partitioned between roles, so no group, scenario, counterfactual group or
domain is shared across roles (validated in data/dataset_validator.py).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from physiworld_uav.config.compose import load_template, to_plain
from physiworld_uav.core.seeds import SeedNode, root
from physiworld_uav.drones.params import DroneParams, check_against_omnidrones
from physiworld_uav.excitation.open_loop import KINDS as OPEN_LOOP_KINDS, make_program
from physiworld_uav.excitation.reference_perturbation import perturb
from physiworld_uav.randomization.domain_generator import generate_domains
from physiworld_uav.randomization.parameter_sampler import DomainSamplerConfig, Domain
from physiworld_uav.trajectories.feasibility import FeasibilityLimits, check
from physiworld_uav.trajectories.library import ATTITUDE_FAMILIES, Anchor, _draw, make_reference
from physiworld_uav.wind.regimes import WindRegime, assign_strata, sample_wind

from .counterfactual import build_siblings

ID_ROLES = ("train", "val_id", "test_id")


class PlanError(ValueError):
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def allocate(weights: Sequence[float], n: int) -> list[int]:
    """Largest-remainder allocation of ``n`` items to ``weights`` (deterministic)."""
    w = np.asarray(weights, dtype=float)
    if n < 0 or w.sum() <= 0 or np.any(w < 0):
        raise PlanError("invalid allocation weights")
    raw = w / w.sum() * n
    base = np.floor(raw).astype(int)
    rem = n - base.sum()
    order = np.argsort(-(raw - base), kind="stable")
    base[order[:rem]] += 1
    return base.tolist()


def _role_of(part: Mapping[str, Any]) -> list[tuple[str, float]]:
    roles = part.get("roles") or {part["role"]: 1.0}
    return [(str(r), float(f)) for r, f in roles.items()]


def _ood_category(role: str) -> str | None:
    return role.split("_ood_", 1)[1] if "_ood_" in role else None


def _apply_controller_randomisation(ctrl: dict[str, Any], rand: Mapping[str, Any], ctrl_cfg: Mapping[str, Any],
                                    rng: np.random.Generator) -> None:
    """Scale the controller gains of one group by random factors (part key ``controller_randomisation``).

    Position mode: ``position_gain`` and ``velocity_gain`` (Lee controller) are the nominal gains times a scale each. Attitude mode: the
    slow position-hold gains ``kp_*`` / ``kd_*`` times ``attitude_hold_scale``. Rate / open-loop / replay modes have nothing to scale.
    The sampled gains are written into the row (``controller.gains`` / ``attitude_hold``) so the episode is reproducible from its row."""
    mode = ctrl["mode"]
    scales: dict[str, float] = {}
    if mode == "position":
        nominal = (ctrl_cfg.get("position") or {}).get("nominal_gains")
        if not nominal:
            raise PlanError("controller_randomisation needs controller.position.nominal_gains in config/controller/default.yaml")
        s_p = float(_draw(rand.get("position_gain_scale", 1.0), rng))
        s_v = float(_draw(rand.get("velocity_gain_scale", 1.0), rng))
        ctrl["gains"] = {"position_gain": [float(x) * s_p for x in nominal["position_gain"]],
                         "velocity_gain": [float(x) * s_v for x in nominal["velocity_gain"]]}
        scales = {"position_gain_scale": s_p, "velocity_gain_scale": s_v}
    elif mode == "attitude" and ctrl.get("attitude_hold"):
        s = float(_draw(rand.get("attitude_hold_scale", 1.0), rng))
        ctrl["attitude_hold"] = {k: (float(v) * s if k in ("kp_xy", "kd_xy", "kp_z", "kd_z") else v)
                                 for k, v in dict(ctrl["attitude_hold"]).items()}
        scales = {"attitude_hold_scale": s}
    ctrl["randomisation"] = scales


def _sample_ic(tmpl: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    pos = tmpl.get("position", {"x": 0.0, "y": 0.0, "z": 2.0})
    return {
        "position": [float(_draw(pos[a], rng)) for a in ("x", "y", "z")],
        "yaw_deg": float(_draw(tmpl.get("yaw_deg", 0.0), rng)),
        "velocity_W": [float(_draw(v, rng)) for v in tmpl.get("velocity_W", [0.0, 0.0, 0.0])],
        "angular_velocity_B": [float(_draw(v, rng)) for v in tmpl.get("angular_velocity_B", [0.0, 0.0, 0.0])],
        "motor": str(tmpl.get("motor", "hover_trim")),
    }


def _seed_record(node: SeedNode) -> dict[str, Any]:
    return {"seed": node.seed, "path": node.path_str}


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    dataset_id: str
    dataset_version: str
    experiment_id: str
    seed: int
    episodes: list[dict[str, Any]]
    parts: list[dict[str, Any]] = field(default_factory=list)

    def by_role(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.episodes:
            out[e["split_role"]] = out.get(e["split_role"], 0) + 1
        return out

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for e in self.episodes:
                f.write(json.dumps(e, sort_keys=True) + "\n")

    @staticmethod
    def read_rows(path: str | Path) -> list[dict[str, Any]]:
        return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def plan_dataset(cfg: Mapping[str, Any]) -> Manifest:
    cfg = to_plain(cfg)
    exp = cfg["experiment"]
    phys = cfg["physics"]
    dt = float(phys["dt"])
    dec = int(phys["control_decimation"])
    model_dt = dt * dec
    nominal = DroneParams.from_dict(cfg["drone"])
    issues = check_against_omnidrones(nominal)
    if issues:
        raise PlanError("drone config disagrees with OmniDrones: " + "; ".join(issues))
    limits = FeasibilityLimits.from_dict(cfg.get("limits", {}).get("feasibility"))
    ctrl_cfg = cfg.get("controller", {})
    seed_root = root(int(exp["seed"]))

    episodes: list[dict[str, Any]] = []
    parts_summary: list[dict[str, Any]] = []
    for part in exp["parts"]:
        part_name = str(part["name"])
        pnode = seed_root.child(f"part:{part_name}")
        roles = _role_of(part)
        n_groups = int(part["n_groups"])
        group_counts = allocate([f for _, f in roles], n_groups)
        if any(c == 0 for c in group_counts):
            raise PlanError(f"part {part_name}: a requested split role would receive 0 groups "
                            f"({dict(zip([r for r, _ in roles], group_counts))}); increase n_groups")

        domain_cfg = load_template("domain", part.get("domain", "nominal"), part.get("domain_overrides"))
        n_dom = int(part.get("n_domains", 1))
        if domain_cfg.get("region", "nominal") != "nominal" and n_dom < len(roles):
            raise PlanError(f"part {part_name}: need ≥ {len(roles)} domains to give each role its own")
        domains = generate_domains(domain_cfg, nominal, n_dom, pnode.child("domains"),
                                   id_prefix=f"{part_name}_dom")
        id_hw = {k: float(v.get("id_halfwidth", 0.0))
                 for k, v in (domain_cfg.get("parameters") or {}).items()}
        if len(domains) == 1:
            role_domains = {r: domains for r, _ in roles}
        else:
            dcounts = allocate([f for _, f in roles], len(domains))
            if any(c == 0 for c in dcounts):
                raise PlanError(f"part {part_name}: too few domains to partition among roles")
            role_domains, k = {}, 0
            for (r, _), c in zip(roles, dcounts):
                role_domains[r], k = domains[k:k + c], k + c

        regime_cfg = load_template("wind/regimes", part["wind_regime"], part.get("wind_overrides"))
        regime = WindRegime.from_dict(regime_cfg)

        mix = part["mix"]
        entry_counts = allocate([float(m.get("weight", 1.0)) for m in mix], n_groups)
        entry_of_group = [i for i, c in enumerate(entry_counts) for _ in range(c)]
        # deterministic interleaving of entries and roles across groups
        order = pnode.child("assignment").rng().permutation(n_groups)
        role_of_group = [r for (r, _), c in zip(roles, group_counts) for _ in range(c)]
        # stratified mean wind speed (regime key ``speed_strata`` = K > 0): within each split role the groups are spread evenly over K
        # equal-width speed bands (a role with 6 groups and K = 5 has one group in five bands and two in one)
        speed_band: dict[int, tuple[int, tuple[float, float]]] = {}
        if regime.speed_strata > 0:
            bands = regime.speed_bands()
            for r, _ in roles:
                gs = [g for g, rr in enumerate(role_of_group) if rr == r]
                for g, b in zip(gs, assign_strata(len(gs), regime.speed_strata, pnode.child("wind_strata").child(f"role:{r}").rng(), regime.speed_strata_weights)):
                    speed_band[g] = (b, bands[b])

        cf_tmpl = None
        if part.get("counterfactual"):
            cf_tmpl = load_template("counterfactual", part["counterfactual"]["template"],
                                    part["counterfactual"].get("overrides"))
            cf_fraction = float(part["counterfactual"].get("fraction", 1.0))

        role_dom_counter = {r: 0 for r, _ in roles}
        for g in range(n_groups):
            entry = mix[entry_of_group[order[g]]]
            role = role_of_group[g]
            gnode = pnode.child(f"group:{g:05d}")
            group_id = f"{part_name}_g{g:05d}"
            pool = role_domains[role]
            domain: Domain = pool[role_dom_counter[role] % len(pool)]
            role_dom_counter[role] += 1
            drone = domain.drone(nominal)

            scen = load_template("scenario", entry["scenario"], entry.get("overrides"))
            ic_t = load_template("initial_conditions", scen.get("initial_conditions", "hover_trim"))
            data_type = str(scen.get("data_type", "closed_loop"))
            duration_raw = float(_draw(scen.get("duration_s", 10.0), gnode.child("duration").rng()))
            n_ctrl = max(1, int(round(duration_raw / model_dt)))
            n_steps = n_ctrl * dec
            duration = n_steps * dt

            # initial conditions + reference / open-loop program (feasibility-checked)
            reference = open_loop = None
            feas = None
            for attempt in range(int(scen.get("max_resample", 20))):
                anode = gnode.child(f"attempt:{attempt}")
                ic = _sample_ic(ic_t, anode.child("initial_conditions").rng())
                exc_tmpl = (load_template("excitation", entry["excitation"])
                            if entry.get("excitation") else {})
                if data_type == "closed_loop":
                    ref = make_reference(scen["family"], scen.get("params", {}), scen.get("yaw", {}),
                                         Anchor(tuple(ic["position"]), math.radians(ic["yaw_deg"])),
                                         duration, anode.child("trajectory").rng())
                    ref = perturb(ref, exc_tmpl.get("channels"), anode.child("excitation").rng())
                    feas = check(ref, drone, limits)
                    if feas.feasible:
                        reference = ref.to_dict()
                        break
                else:
                    kind = str(scen["open_loop_kind"])
                    if kind not in OPEN_LOOP_KINDS:
                        raise PlanError(f"scenario {scen['name']}: unknown open_loop_kind {kind!r}")
                    prog = make_program(kind, {**scen.get("channels", {}), **exc_tmpl.get("channels", {})},
                                        drone, anode.child("excitation").rng(), duration)
                    open_loop = prog.to_dict()
                    break
            else:
                raise PlanError(f"group {group_id} ({scen['name']}): no feasible reference after "
                                f"{scen.get('max_resample', 20)} attempts; last: {feas.violations if feas else '-'}")

            wind = sample_wind(regime, gnode.child("wind").rng(), duration=duration, dt=dt,
                               profile=entry.get("wind_profile"), speed_range=speed_band[g][1] if g in speed_band else None)
            if data_type == "closed_loop":
                controller_mode = "attitude" if scen["family"] in ATTITUDE_FAMILIES else "position"
            else:
                controller_mode = {"rotor_command": "rotor_command", "collective_rate": "rate",
                                   "body_wrench": "rotor_command"}[open_loop["kind"]]
            base = {
                "dataset_id": exp["dataset_id"], "experiment_id": exp["name"],
                "part": part_name, "group_id": group_id,
                "scenario_id": f"{part_name}_scn_{g:05d}",
                "scenario_template": scen["name"], "mode": scen.get("mode"),
                "experiment_group": entry.get("experiment_group"),
                "split_role": role, "ood_category": _ood_category(role), "data_type": data_type,
                "duration_s": duration, "n_steps": n_steps,
                "sim": {"physics_dt": dt, "control_decimation": dec, "model_dt": model_dt,
                        "imu_transport_correction": bool(phys.get("imu_transport_correction", False)),
                        "settle_steps": int(phys.get("settle_steps", 1))},
                "domain": {**domain.to_dict(), "drone": drone.summary()},
                "controller": {"mode": controller_mode,
                               "policy": str(ctrl_cfg.get("policy", "true")),
                               "gains": ctrl_cfg.get(controller_mode, {}).get("gains")
                               if isinstance(ctrl_cfg.get(controller_mode), Mapping) else None,
                               "attitude_hold": ctrl_cfg.get("attitude", {}).get("attitude_hold")
                               if controller_mode == "attitude" else None,
                               "hold_position": bool(scen.get("hold_position", True))},
                "forces": {"name": str(part.get("forces", "none")),
                           "config": load_template("forces", str(part.get("forces", "none")))},
                "initial_conditions": ic, "reference": reference, "open_loop": open_loop,
                "feasibility": None if feas is None else feas.peaks,
                "wind": {**wind.to_dict(), "limits": regime_cfg["limits"],
                         **({"speed_stratum": {"index": speed_band[g][0], "of": regime.speed_strata, "range_mps": list(speed_band[g][1])}}
                            if g in speed_band else {})},
                "excitation": entry.get("excitation"),
                "seeds": {"group": _seed_record(gnode),
                          "initial_conditions": _seed_record(anode.child("initial_conditions")),
                          "trajectory": _seed_record(anode.child("trajectory")),
                          "excitation": _seed_record(anode.child("excitation")),
                          "wind": {"seed": wind.seed, "path": gnode.child("wind").path_str},
                          "domain": domain.seed_path},
                "counterfactual": None, "replay_of": None,
            }

            if part.get("controller_randomisation"):
                _apply_controller_randomisation(base["controller"], part["controller_randomisation"], ctrl_cfg,
                                                gnode.child("controller_gains").rng())

            use_cf = cf_tmpl is not None and (
                cf_fraction >= 1.0 or gnode.child("cf_select").rng().random() < cf_fraction)
            if use_cf:
                sibs = build_siblings(
                    base, cf_tmpl, duration=duration, dt=dt, group_id=group_id, id_halfwidths=id_hw,
                    drone_summary=lambda dom: Domain(
                        domain_id=dom["domain_id"], region=dom["region"], factors=dom["factors"],
                    ).drone(nominal).summary(),
                    forces_loader=lambda name: load_template("forces", name))
            else:
                sibs = [base]
            for i, s in enumerate(sibs):
                if g in speed_band:                       # the siblings' wind dicts are rebuilt by the counterfactual builder: keep the stratum on every row
                    s["wind"] = {**s["wind"], "speed_stratum": base["wind"]["speed_stratum"]}
                s["episode_id"] = f"{group_id}_e{i}"
                s["seeds"] = {**s["seeds"], "episode": seed_root.child(s["episode_id"]).seed}
            if use_cf and cf_tmpl.get("mode") == "fixed_action":
                for s in sibs[1:]:
                    s["replay_of"] = sibs[0]["episode_id"]
            episodes.extend(sibs)
        parts_summary.append({"name": part_name, "roles": dict(roles), "n_groups": n_groups,
                              "n_domains": len(domains), "wind_regime": regime.name,
                              "domain": domain_cfg.get("name"), "mix": mix})
    return Manifest(dataset_id=str(exp["dataset_id"]), dataset_version=str(exp["dataset_version"]),
                    experiment_id=str(exp["name"]), seed=int(exp["seed"]),
                    episodes=episodes, parts=parts_summary)
