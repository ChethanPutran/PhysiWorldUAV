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
    check_guards(row)                                            # truth_default was approved (Q3): accepted
    unvalidated = copy.deepcopy(row)
    unvalidated["forces"]["config"]["validated"] = False
    with pytest.raises(RunnerGuardError):
        check_guards(unvalidated)
    check_guards(unvalidated, allow_unvalidated=True)            # dev flag only


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
    row["forces"] = {"name": "truth_default", "config": {**load_template("forces", "truth_default"),
                                                         "validated": False}}      # exercise the unvalidated path
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


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_replay_of_short_baseline_is_truncated_and_flagged(pilot):
    cfg, manifest = pilot
    row = next(r for r in manifest.episodes if r.get("replay_of"))
    D = int(row["sim"]["control_decimation"])
    nominal = DroneParams.from_dict(cfg["drone"])
    hover = np.tile(nominal.hover_command.reshape(1, -1), (row["n_steps"] // D, 1)).astype(np.float32)

    def run(cmds):
        return run_episode(MockSim(nominal, cfg["physics"]["dt"]), row, nominal, termination=Termination(),
                           config_hash="h", dataset_version="0", replay_commands=cmds)

    _, full = run(hover)
    assert full.n_steps == row["n_steps"] and full.stats["replay_truncated_to_baseline"] is False
    _, short = run(hover[:40])                                   # baseline ended after 40 control steps
    assert short.n_steps == 40 * D and short.stats["replay_truncated_to_baseline"] is True
    with pytest.raises(ValueError):
        run(hover[:0])


def test_index_records_failures_and_is_incomplete(tmp_path, pilot):
    from physiworld_uav.experiments.generate import _write_index

    class Report:
        ok = True
        hard: list = []
        soft: list = []

    _, manifest = pilot
    _write_index(tmp_path, manifest, [], Report(), 1.0, [{"episode_id": "x", "error": "ValueError", "message": "m",
                                                          "traceback": "tb"}])
    idx = json.loads((tmp_path / "index.json").read_text())
    assert idx["complete"] is False and idx["failures"][0]["episode_id"] == "x"


def _mock_episode(row_index=0):
    cfg = to_plain(compose(["experiment=smoke"]))
    row = plan_dataset(cfg).episodes[row_index]
    nominal = DroneParams.from_dict(cfg["drone"])
    arrays, meta = run_episode(MockSim(nominal, cfg["physics"]["dt"]), row, nominal, termination=Termination(),
                               config_hash="h", dataset_version="0")
    return arrays, meta


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_scan_shards_rebuilds_rows_and_drops_interrupted_writes(tmp_path):
    import h5py
    from physiworld_uav.data.writer import ShardWriter
    from physiworld_uav.experiments.generate import _scan_shards
    arrays, meta = _mock_episode()
    with ShardWriter(tmp_path / "shard_000.h5") as w:
        w.write(arrays, meta)
    with h5py.File(tmp_path / "shard_000.h5", "a") as f:               # a write killed before its metadata
        f["episodes"].create_group("half_written").create_dataset("x", data=[1.0])
    rows, where = _scan_shards(tmp_path, {meta.episode_id: 0})
    assert [r["episode_id"] for r in rows] == [meta.episode_id] and where == {meta.episode_id: "shard_000.h5"}
    assert rows[0]["shard"] == "shard_000.h5"
    with h5py.File(tmp_path / "shard_000.h5", "r") as f:
        assert "half_written" not in f["episodes"]


def test_checkpoint_index_is_atomic_and_marks_in_progress(tmp_path, pilot):
    from physiworld_uav.experiments.generate import _write_index
    _, manifest = pilot
    _write_index(tmp_path, manifest, [], None, 1.0, [], in_progress=True)
    idx = json.loads((tmp_path / "index.json").read_text())
    assert idx["in_progress"] is True and idx["complete"] is False and idx["validation"]["ok"] is None
    assert not (tmp_path / "index.json.tmp").exists()


# -- parallel generation: worker assignment, shard isolation, merge ----------------------------
from physiworld_uav.experiments import generate as G


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_worker_assignment_partitions_rows_keeps_groups_together_and_balances(pilot, n):
    cfg, m = pilot
    rows = m.episodes
    owner = G.assign_workers(rows, n)
    assert owner == G.assign_workers(rows, n)                                    # deterministic
    parts = [G.worker_rows(rows, w, n) for w in range(n)]
    ids = [r["episode_id"] for p in parts for r in p]
    assert sorted(ids) == sorted(r["episode_id"] for r in rows) and len(ids) == len(set(ids))   # exact partition
    by_group: dict[str, set[int]] = defaultdict(set)
    for w, p in enumerate(parts):
        for r in p:
            by_group[r["group_id"]].add(w)
    assert all(len(ws) == 1 for ws in by_group.values())                         # a counterfactual group never splits
    where = {r["episode_id"]: w for w, p in enumerate(parts) for r in p}
    assert all(where[r["replay_of"]] == where[r["episode_id"]] for r in rows if r.get("replay_of"))   # Mode B reads its baseline
    cost = [sum(r["n_steps"] for r in p) for p in parts]
    biggest_group = max(sum(r["n_steps"] for r in rows if r["group_id"] == g) for g in by_group)
    assert max(cost) - min(cost) <= biggest_group                                # LPT bound
    if n > 1:
        assert min(len(p) for p in parts) > 0


def test_worker_shard_names_and_scan_isolation(tmp_path):
    import h5py
    assert G.shard_prefix(None) == "shard_" and G.shard_prefix((0, 1)) == "shard_" and G.shard_prefix((3, 8)) == "shard_w03_"
    for name in ("shard_w00_000.h5", "shard_w01_000.h5"):
        with h5py.File(tmp_path / name, "w") as f:
            f.require_group("episodes").require_group("half_written")            # no metadata attribute: an interrupted write
    rows, _ = G._scan_shards(tmp_path, {}, "shard_w00_")                         # worker 0 cleans only its own file
    assert rows == []
    with h5py.File(tmp_path / "shard_w00_000.h5") as f0, h5py.File(tmp_path / "shard_w01_000.h5") as f1:
        assert "half_written" not in f0["episodes"] and "half_written" in f1["episodes"]
    G._scan_shards(tmp_path, {})                                                 # a merge sees every worker's shards
    with h5py.File(tmp_path / "shard_w01_000.h5") as f1:
        assert "half_written" not in f1["episodes"]


class _ClosableMock(MockSim):
    def close(self):
        pass


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_sharded_run_and_merge_equal_the_single_process_dataset(tmp_path, monkeypatch):
    def overrides(tag):
        return ["experiment=smoke", "experiment.parts.0.n_groups=4", f"experiment.dataset_version={tag}",
                f"dataset.output_root={tmp_path}"]

    def fake_make_sim(phys, first):
        return _ClosableMock(DroneParams.from_dict(to_plain(compose(["experiment=smoke"]))["drone"]), phys["dt"])

    monkeypatch.setattr(G, "_make_sim", fake_make_sim)
    monkeypatch.setattr(G, "_sim_settings", lambda sim: {})
    cfg_par = compose(overrides("par"))
    out = G.dataset_dir(to_plain(cfg_par))
    for w in (1, 0):                                                              # order of the workers does not matter
        G.run(cfg_par, shard=(w, 2))
    assert not (out / "index.json").exists()                                     # workers never write the final index
    assert (out / "index.worker00.json").exists() and (out / "index.worker01.json").exists()
    assert sorted(p.name for p in out.glob("shard_*.h5")) == ["shard_w00_000.h5", "shard_w01_000.h5"]
    G.merge(cfg_par)
    idx = json.loads((out / "index.json").read_text())
    assert idx["n_written"] == idx["n_planned"] == 4 and idx["validation"]["ok"] and idx["complete"]
    assert len({r["episode_id"] for r in idx["episodes"]}) == 4
    # one process, no shards: identical episodes (the worker split never changes an episode)
    cfg_one = compose(overrides("one"))
    G.run(cfg_one)
    one = G.dataset_dir(to_plain(cfg_one))
    from physiworld_uav.data.writer import read_episode
    rows_one = {r["episode_id"]: r["shard"] for r in json.loads((one / "index.json").read_text())["episodes"]}
    for r in idx["episodes"]:
        a, _ = read_episode(out / r["shard"], r["episode_id"]); b, _ = read_episode(one / rows_one[r["episode_id"]], r["episode_id"])
        assert np.array_equal(a["actions/rotor_cmd"], b["actions/rotor_cmd"])
        assert np.array_equal(a["ground_truth/position_W"], b["ground_truth/position_W"])


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_worker_resume_and_merge_refuses_in_progress_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "_make_sim", lambda phys, first: _ClosableMock(
        DroneParams.from_dict(to_plain(compose(["experiment=smoke"]))["drone"]), phys["dt"]))
    monkeypatch.setattr(G, "_sim_settings", lambda sim: {})
    cfg = compose(["experiment=smoke", "experiment.parts.0.n_groups=4", "experiment.dataset_version=res",
                   f"dataset.output_root={tmp_path}"])
    out = G.dataset_dir(to_plain(cfg))
    G.run(cfg, shard=(0, 2), limit=1)                                             # "crash" after one episode
    w0 = json.loads((out / "index.worker00.json").read_text())
    assert w0["n_written"] == 1 and w0["n_assigned"] == 2 and not w0["in_progress"]
    with pytest.raises(SystemExit):                                               # a second start without --resume is refused
        G.run(cfg, shard=(0, 2))
    G.run(cfg, shard=(0, 2), resume=True)
    assert json.loads((out / "index.worker00.json").read_text())["n_written"] == 2
    G.run(cfg, shard=(1, 2))
    (out / "index.worker01.json").write_text(json.dumps({**json.loads((out / "index.worker01.json").read_text()), "in_progress": True}))
    with pytest.raises(SystemExit, match="in_progress"):
        G.merge(cfg)
    G.merge(cfg, force=True)
    assert json.loads((out / "index.json").read_text())["n_written"] == 4


# -- IMU transport correction in the runner ----------------------------------------------------------
class _ArtefactMock(MockSim):
    """Mock whose IMU reports specific force minus ω_B × v_B (the Isaac artefact), flying with a constant velocity and rotation rate
    (attitude frozen at the episode's initial yaw: body axes differ from world axes)."""

    def __init__(self, drone, dt, artefact: bool):
        super().__init__(drone, dt)
        self.artefact = artefact

    def reset_episode(self, position, quat_wxyz, velocity_W, angular_velocity_W, throttle):
        super().reset_episode(position, quat_wxyz, [2.0, 0.0, 0.0], [0.0, 0.4, 0.0], throttle)

    def read_imu(self):
        acc, gyro, ok = super().read_imu()
        if self.artefact:                 # body-frame transport term, computed here from the quaternion (independent of the code under test)
            w, x, y, z = (float(c) for c in self.q)
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])      # body -> world
            acc = acc - np.cross(R.T @ self.w.numpy(), R.T @ self.v.numpy())
        return acc, gyro, ok

    def close(self):
        pass


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_runner_removes_the_imu_transport_artefact_when_configured():
    cfg = to_plain(compose(["experiment=smoke"]))
    row = plan_dataset(cfg).episodes[0]
    assert row["sim"]["imu_transport_correction"] is True                         # default of config/physics for new datasets
    nominal = DroneParams.from_dict(cfg["drone"])

    def run(artefact, corrected):
        r = copy.deepcopy(row); r["sim"]["imu_transport_correction"] = corrected
        sim = _ArtefactMock(nominal, cfg["physics"]["dt"], artefact)
        return run_episode(sim, r, nominal, termination=Termination(), config_hash="h", dataset_version="0", enforce_guards=False)

    clean, m_clean = run(artefact=False, corrected=False)                           # an ideal sensor
    raw, m_raw = run(artefact=True, corrected=False)                                # Isaac as in the pilot
    fixed, m_fix = run(artefact=True, corrected=True)                               # Isaac + the correction
    a_c, a_r, a_f = (x["observations/imu_accel_B"][1:] for x in (clean, raw, fixed))
    assert np.abs(a_r - a_c).max() > 0.5                                            # the artefact is visible (|ω×v| = 0.8 m/s²)
    assert np.allclose(a_f, a_c, atol=1e-4)                                         # the correction restores the specific force
    assert m_fix.stats["imu_transport_corrected"] is True and m_raw.stats["imu_transport_corrected"] is False
    for k in ("ground_truth/position_W", "ground_truth/velocity_W", "actions/rotor_cmd"):
        assert np.array_equal(clean[k], fixed[k])                                   # only the logged accelerometer changes


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_first_imu_sample_is_the_analytic_rest_value_not_the_sensor_read():
    """The t = 0 IMU sample depends on no simulator history: R(q0)ᵀ(0, 0, g) with zero gyro, whatever the sensor reports."""
    from physiworld_uav.drones.params import G
    from physiworld_uav.experiments.runner import _quat_from_yaw
    import math
    cfg = to_plain(compose(["experiment=smoke"]))
    row = plan_dataset(cfg).episodes[0]
    nominal = DroneParams.from_dict(cfg["drone"])

    class Junk(MockSim):
        first = True
        def read_imu(self):
            if Junk.first:                      # the settle-step read: pretend it carries leftover state
                Junk.first = False
                return np.array([4.6e-3, -1.3e-3, 9.8]), np.array([0.1, 0.2, 0.3]), True
            return super().read_imu()
        def close(self): pass

    Junk.first = True
    arrays, meta = run_episode(Junk(nominal, cfg["physics"]["dt"]), row, nominal, termination=Termination(), config_hash="h",
                               dataset_version="0", enforce_guards=False)
    q = _quat_from_yaw(math.radians(row["initial_conditions"]["yaw_deg"]))
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    assert np.allclose(arrays["observations/imu_accel_B"][0], R.T @ np.array([0, 0, G]), atol=1e-5)
    assert np.allclose(arrays["observations/imu_gyro_B"][0], 0.0) and bool(arrays["observations/imu_valid"][0])


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_runner_reports_progress_every_control_interval():
    cfg = to_plain(compose(["experiment=smoke"]))
    row = plan_dataset(cfg).episodes[0]
    nominal = DroneParams.from_dict(cfg["drone"])
    seen = []
    arrays, meta = run_episode(MockSim(nominal, cfg["physics"]["dt"]), row, nominal, termination=Termination(), config_hash="h", dataset_version="0",
                               progress=lambda k, n: seen.append((k, n)))
    D = row["sim"]["control_decimation"]
    assert [k for k, _ in seen] == list(range(0, meta.n_steps, D)) and {n for _, n in seen} == {meta.n_steps}


@pytest.mark.skipif(not omni_drones_available(), reason="omni_drones not importable")
def test_worker_writes_a_progress_file(tmp_path, monkeypatch):
    def fake_make_sim(phys, first):
        return _ClosableMock(DroneParams.from_dict(to_plain(compose(["experiment=smoke"]))["drone"]), phys["dt"])
    monkeypatch.setattr(G, "_make_sim", fake_make_sim)
    monkeypatch.setattr(G, "_sim_settings", lambda sim: {})
    cfg = compose(["experiment=smoke", "experiment.parts.0.n_groups=2", "experiment.dataset_version=prog", f"dataset.output_root={tmp_path}"])
    out = G.dataset_dir(to_plain(cfg))
    G.run(cfg, shard=(0, 1)) if False else G.run(cfg, shard=(0, 2))
    pf = out / "progress.worker00.json"
    assert pf.exists() and not list(out.glob("progress*.tmp"))
    d = json.loads(pf.read_text())
    assert d["worker"] == 0 and d["episode_id"] and d["n_steps"] > 0 and 0 <= d["step"] <= d["n_steps"] and d["episodes_timed"] >= 1 and d["mean_episode_s"] is not None
