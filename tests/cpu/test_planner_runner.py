"""Config composition, experiment planning, counterfactuals and the runner (mock sim)."""
from __future__ import annotations

import copy
import json
from collections import defaultdict

import numpy as np
import pytest
import torch

from physiworld_uav.config import compose, to_plain
from physiworld_uav.data.dataset_validator import validate_episode
from physiworld_uav.drones.params import DroneParams
from physiworld_uav.experiments.counterfactual import plan_diff
from physiworld_uav.experiments.plan import allocate, plan_dataset
from physiworld_uav.experiments.runner import RunnerGuardError, Termination, check_guards, run_episode

from conftest import omni_drones_available


@pytest.fixture(scope="module")
def pilot():
    cfg = to_plain(compose(["experiment=pilot"]))
    return cfg, plan_dataset(cfg)


def test_allocate_is_exact_and_deterministic():
    assert allocate([0.7, 0.15, 0.15], 40) == [28, 6, 6]
    assert sum(allocate([1, 2, 3], 7)) == 7


def test_plan_is_deterministic(pilot):
    cfg, m = pilot
    m2 = plan_dataset(copy.deepcopy(cfg))
    assert json.dumps(m.episodes, sort_keys=True) == json.dumps(m2.episodes, sort_keys=True)


def test_plan_parts_are_seed_isolated(pilot):
    cfg, m = pilot
    cfg2 = copy.deepcopy(cfg)
    for p in cfg2["experiment"]["parts"]:
        if p["name"] == "ood_mag":
            p["n_groups"] = 12
    m2 = plan_dataset(cfg2)
    a = [e for e in m.episodes if e["part"] == "id_cl"]
    b = [e for e in m2.episodes if e["part"] == "id_cl"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_split_roles_are_group_and_domain_disjoint(pilot):
    _, m = pilot
    roles_of = defaultdict(set)
    for e in m.episodes:
        roles_of[("group", e["group_id"])].add(e["split_role"])
        roles_of[("scenario", e["scenario_id"])].add(e["split_role"])
        if e["domain"]["region"] != "nominal" and not e["domain"]["domain_id"].endswith(("_cf1", "_cf2")):
            roles_of[("domain", e["domain"]["domain_id"])].add(e["split_role"])
    assert all(len(v) == 1 for v in roles_of.values())


def test_roles_get_the_intended_wind_and_physics(pilot):
    _, m = pilot
    for e in m.episodes:
        role, cat, region = e["split_role"], e["wind"]["category"], e["domain"]["region"]
        if role in ("train", "val_id", "test_id"):
            assert cat == "id" and region in ("id", "nominal")
            assert e["wind"]["stats"]["peak_speed"] <= 4.0 + 1e-9
        elif role.endswith("ood_magnitude"):
            assert cat == "ood_magnitude" and 4.0 < e["wind"]["stats"]["mean_horizontal_speed"] <= 8.0
        elif role.endswith("ood_physics"):
            assert region in ("ood_physics", "ood_combination")
        elif role.endswith("ood_trajectory"):
            assert e["scenario_template"] in ("helix", "figure_eight")
    id_templates = {e["scenario_template"] for e in m.episodes if e["split_role"] in ("train", "val_id", "test_id")}
    assert not id_templates & {"helix", "figure_eight"}                   # Q8 holdout


def test_counterfactual_groups_change_one_variable(pilot):
    _, m = pilot
    groups = defaultdict(list)
    for e in m.episodes:
        if e["counterfactual"]:
            groups[e["counterfactual"]["group_id"]].append(e)
    assert groups
    for gid, sibs in groups.items():
        base = [s for s in sibs if s["counterfactual"]["is_baseline"]]
        assert len(base) == 1
        var = base[0]["counterfactual"]["variable"]
        prefix = {"wind_magnitude": "wind", "mass": "domain"}[var]
        for s in sibs:
            assert {k.split("/")[0] for k in plan_diff(base[0], s)} <= {prefix}
            assert s["reference"] == base[0]["reference"] and s["open_loop"] == base[0]["open_loop"]
            assert s["initial_conditions"] == base[0]["initial_conditions"]
            if s["counterfactual"]["mode"] == "fixed_action" and not s["counterfactual"]["is_baseline"]:
                assert s["replay_of"] == base[0]["episode_id"] and s["controller"]["mode"] == "replay"
        if var == "wind_magnitude":
            assert base[0]["wind"]["stats"]["peak_speed"] == 0.0              # D8 zero-wind baseline


def test_runner_guards(pilot):
    _, m = pilot
    row = next(e for e in m.episodes if e["forces"]["name"] != "none")
    with pytest.raises(RunnerGuardError):
        check_guards(row)


# -- runner with a mock simulator -------------------------------------------------------

class MockSim:
    """Rigid point mass with the OmniDrones motor model and thrust along world z
    (attitude frozen). Enough to exercise plumbing, timing and schema — not physics."""

    def __init__(self, drone: DroneParams, dt: float):
        self.device = torch.device("cpu")
        self.d, self.dt = drone, dt
        self.kf = torch.tensor(drone.max_thrust_per_rotor, dtype=torch.float32)
        self._step = 0
        self._ext = (torch.zeros(1, 3), torch.zeros(1, 3))

    def reset_episode(self, position, quat_wxyz, velocity_W, angular_velocity_W, throttle):
        self.p = torch.tensor(position, dtype=torch.float32)
        self.q = torch.tensor(quat_wxyz, dtype=torch.float32)
        self.v = torch.tensor(velocity_W, dtype=torch.float32)
        self.w = torch.tensor(angular_velocity_W, dtype=torch.float32)
        self.thr = torch.tensor(throttle, dtype=torch.float32)

    def get_state(self):
        return torch.cat([self.p, self.q, self.v, self.w]).reshape(1, 1, 13)

    def throttle(self):
        return self.thr.reshape(1, -1)

    def apply_rotor_command(self, cmd):
        target = torch.sqrt(((cmd.reshape(-1) + 1) / 2).clamp(0, 1))
        self.thr = self.thr + 0.43 * (target - self.thr)
        self.F = float((self.thr ** 2 * self.kf).sum())

    def step(self):
        a = torch.tensor([0, 0, self.F / self.d.mass - 9.81]) + self._ext[0].reshape(3) / self.d.mass
        self.v = self.v + a * self.dt
        self.p = self.p + self.v * self.dt
        self._step += 1

    def read_imu(self):
        return np.array([0.0, 0.0, self.F / self.d.mass]), np.zeros(3), True

    def bind_domain(self, params, dt):
        self.d = params
        return {"total_mass_kg": params.mass, "base_inertia_diag_kgm2": list(params.inertia_diag),
                "composite_inertia_effective_kgm2": np.diag(params.inertia_diag).tolist(),
                "force_constants": list(params.force_constants),
                "moment_constants": list(params.moment_constants),
                "motor_time_constant_s": params.effective_motor_time_constant, "drag_coef_omnidrones": 0.0}

    def set_external_wrench(self, f, t):
        self._ext = (f.reshape(1, 3).clone(), t.reshape(1, 3).clone())

    def last_external_wrench(self):
        return self._ext


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_runner_produces_valid_schema_with_mock():
    cfg = to_plain(compose(["experiment=smoke"]))
    row = plan_dataset(cfg).episodes[0]
    nominal = DroneParams.from_dict(cfg["drone"])
    sim = MockSim(nominal, cfg["physics"]["dt"])
    arrays, meta = run_episode(sim, row, nominal, termination=Termination(), config_hash="h",
                               dataset_version="0")
    r = validate_episode(arrays, meta)
    assert r.ok, r.hard
    assert meta.status == "completed" and meta.n_steps == row["n_steps"]
    cmd = arrays["actions/rotor_cmd"]
    D = row["sim"]["control_decimation"]
    assert np.allclose(cmd.reshape(-1, D, 4), cmd[::D][:, None, :])          # held over control intervals
    assert abs(arrays["ground_truth/position_W"][-1, 2] - row["initial_conditions"]["position"][2]) < 0.05
    assert np.allclose(arrays["ground_truth/rotor_speed"][0], nominal.hover_rotor_speed, rtol=1e-5)


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_runner_with_wind_and_forces_logs_consistent_wrench():
    """Unvalidated force config allowed explicitly; wind pushes the (frozen-attitude) mock body."""
    from physiworld_uav.config import load_template
    from physiworld_uav.wind.regimes import WindRegime, sample_wind
    cfg = to_plain(compose(["experiment=smoke"]))
    row = copy.deepcopy(plan_dataset(cfg).episodes[0])
    reg = WindRegime.from_dict(load_template("wind/regimes", "id"))
    w = sample_wind(reg, np.random.default_rng(0), duration=row["duration_s"], dt=row["sim"]["physics_dt"],
                    profile="steady")
    row["wind"] = {**w.to_dict(), "limits": load_template("wind/regimes", "id")["limits"]}
    row["forces"] = {"name": "truth_default", "config": load_template("forces", "truth_default")}
    nominal = DroneParams.from_dict(cfg["drone"])
    with pytest.raises(RunnerGuardError):
        check_guards(row)
    arrays, meta = run_episode(MockSim(nominal, cfg["physics"]["dt"]), row, nominal, termination=Termination(),
                               config_hash="h", dataset_version="0", allow_unvalidated_forces=True)
    r = validate_episode(arrays, meta)
    assert len(r.hard) == 1 and "unvalidated force config" in r.hard[0], r.hard   # never a valid dataset
    tot = arrays["ground_truth/wrench/external_force_W"]
    parts = arrays["ground_truth/wrench/rotor_drag_force_W"] + arrays["ground_truth/wrench/body_drag_force_W"]
    assert np.allclose(tot, parts, atol=1e-6)
    wind_dir = np.asarray(w.spec["steady"]["direction"])[:2]
    assert tot[0, :2] @ wind_dir > 0                                   # drag pushes downwind at rest
    assert meta.forces["applied"] and meta.drone_realised["total_mass_kg"] == pytest.approx(nominal.mass)
