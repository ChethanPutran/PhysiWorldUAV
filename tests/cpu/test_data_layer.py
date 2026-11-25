"""Schema, HDF5 round trip, decimation, model-ready samples, validator (plan §6.5)."""
from __future__ import annotations

import numpy as np
import pytest

from physiworld_uav.data.dataset import ModelReadyDataset
from physiworld_uav.data.dataset_builder import BuilderConfig, to_model_episode
from physiworld_uav.data.dataset_validator import validate_dataset, validate_episode
from physiworld_uav.data.metadata import CounterfactualInfo, EpisodeMetadata
from physiworld_uav.data.resampler import DecimationError, decimate_episode
from physiworld_uav.data.schema import AUXILIARY_PREFIXES, BY_PATH, FIELDS, validate_arrays
from physiworld_uav.data.splits import nested_group_subsets
from physiworld_uav.data.writer import SchemaError, ShardWriter, read_episode

DT, D, N = 0.005, 4, 80


def synthetic_episode(n=N, data_type="closed_loop", seed=0):
    """Physically simple but schema-complete episode: hover with a slow drift."""
    rng = np.random.default_rng(seed)
    t = np.arange(n + 1) * DT
    a: dict[str, np.ndarray] = {}
    for f in FIELDS:
        if f.path.startswith("control/") and data_type != "closed_loop":
            continue
        rows = n + 1 if f.sem == "inst" else n
        if f.dtype == "bool":
            a[f.path] = np.ones((rows,) + f.width, dtype=bool)
            continue
        a[f.path] = (0.01 * rng.standard_normal((rows,) + f.width)).astype(f.dtype)
    a["time/t"] = t
    a["ground_truth/orientation_wxyz"] = np.tile([1.0, 0, 0, 0], (n + 1, 1)).astype(np.float32)
    a["actions/saturated"][:] = False
    a["environment/wind_W"][:] = 0.0
    for k in a:                                                  # force accounting: zero wrench everywhere
        if k.startswith("ground_truth/wrench/"):
            a[k][:] = 0.0
    held = [p for p in a if p.startswith(("actions/", "control/")) and BY_PATH[p].dtype != "bool"]
    for p in held:                                   # hold over control intervals
        a[p] = np.repeat(a[p][::D], D, axis=0)[:n]
    return a


def meta(eid="ep0", role="train", group="g0", scenario="s0", **kw):
    base = dict(dataset_id="t", dataset_version="0", experiment_id="x", episode_id=eid,
                scenario_id=scenario, domain_id="nominal", group_id=group, split_role=role,
                data_type="closed_loop", n_steps=N, config_hash="abc",
                sim={"physics_dt": DT, "control_decimation": D, "model_dt": DT * D},
                seeds={"episode": 1, "initial_conditions": {"seed": 2}, "trajectory": {"seed": hash(eid) % 997},
                       "wind": {"seed": 4}, "excitation": {"seed": 5}},
                drone={"mass_kg": 0.716, "inertia_diag_kgm2": [0.007, 0.007, 0.012], "force_constants": [1] * 4},
                drone_realised={"total_mass_kg": 0.716})
    base.update(kw)
    return EpisodeMetadata(**base)


def test_schema_accepts_synthetic_and_rejects_bad_shapes():
    a = synthetic_episode()
    assert validate_arrays(a, data_type="closed_loop") == []
    b = dict(a)
    b["actions/rotor_cmd"] = b["actions/rotor_cmd"][:-1]
    assert any("shape" in i for i in validate_arrays(b, data_type="closed_loop"))
    c = {k: v for k, v in a.items() if not k.startswith("control/")}
    assert validate_arrays(c, data_type="open_loop") == []


def test_hdf5_roundtrip_and_schema_enforcement(tmp_path):
    a, m = synthetic_episode(), meta()
    with ShardWriter(tmp_path / "s.h5") as w:
        w.write(a, m)
        with pytest.raises(SchemaError):
            w.write(a, m)                                  # duplicate id
        bad = dict(a)
        bad["time/t"] = bad["time/t"][:-2]
        with pytest.raises(SchemaError):
            w.write(bad, meta("ep1"))
    back, m2 = read_episode(tmp_path / "s.h5", "ep0")
    assert m2.to_dict() == m.to_dict()
    for k in a:
        assert np.array_equal(back[k], a[k].astype(BY_PATH[k].dtype))


def test_decimation_semantics():
    a = synthetic_episode()
    d = decimate_episode(a, D)
    m = N // D
    assert d["time/t"].shape == (m + 1,) and np.allclose(np.diff(d["time/t"]), DT * D)
    assert np.array_equal(d["ground_truth/position_W"], a["ground_truth/position_W"][::D])
    assert np.array_equal(d["actions/rotor_cmd"], a["actions/rotor_cmd"][::D])
    mean = a["ground_truth/rotor_speed"][1:].reshape(m, D, 4).mean(axis=1)
    assert np.allclose(d["ground_truth/rotor_speed__mean"], mean, atol=1e-6)
    bad = dict(a)
    bad["actions/rotor_cmd"] = bad["actions/rotor_cmd"] + np.arange(N)[:, None] * 1e-3
    with pytest.raises(DecimationError):
        decimate_episode(bad, D)
    with pytest.raises(DecimationError):
        decimate_episode({k: v[:-1] if BY_PATH[k].sem == "inst" else v[:-1] for k, v in a.items()}, D)


def test_model_ready_sample_semantics():
    cfg = BuilderConfig(decimation=D, history=3, horizon=4, action_key="rotor_cmd")
    ep = to_model_episode(synthetic_episode(), meta(), cfg)
    assert ep.states.shape == (N // D + 1, 13) and ep.actions.shape == (N // D, 4)
    ds = ModelReadyDataset([ep], cfg)
    assert len(ds) == (N // D - 4) - 3 + 1
    s = ds[0]
    t = s["time_index"]
    assert t == 3
    assert np.array_equal(s["state_history"].numpy(), ep.states[t - 2: t + 1])
    assert np.array_equal(s["action_history"].numpy(), ep.actions[t - 3: t])
    assert np.array_equal(s["future_actions"].numpy(), ep.actions[t: t + 4])
    assert np.array_equal(s["future_states"].numpy(), ep.states[t + 1: t + 5])
    assert s["obs_history"].shape == (3, 6)
    assert not any(k.startswith(AUXILIARY_PREFIXES) for k in s)
    # layout p | v | q | ω_B
    a = synthetic_episode()
    assert np.allclose(ep.states[1, 3:6], a["ground_truth/velocity_W"][D])
    assert np.allclose(ep.states[1, 6:10], a["ground_truth/orientation_wxyz"][D])


def test_validator_episode_checks():
    assert validate_episode(synthetic_episode(), meta()).ok
    a = synthetic_episode()
    a["ground_truth/position_W"][5, 0] = np.nan
    assert not validate_episode(a, meta()).ok
    a = synthetic_episode()
    a["observations/imu_valid"][:] = False
    r = validate_episode(a, meta())
    assert any("IMU" in h for h in r.hard)
    a = synthetic_episode()
    a["ground_truth/orientation_wxyz"][10] *= -1
    assert any("sign" in h for h in validate_episode(a, meta()).hard)


def test_validator_leakage_and_counterfactual_rules():
    ok = [meta("a", "train", "g0", "s0"), meta("b", "test_id", "g1", "s1")]
    assert validate_dataset(ok).ok
    leak = [meta("a", "train", "g0", "s0"), meta("b", "test_id", "g0", "s1")]
    assert any("group_id" in h for h in validate_dataset(leak).hard)
    cf = lambda i, base, var="wind_magnitude": CounterfactualInfo("cf0", "closed_loop", var, i, base)  # noqa: E731
    two_base = [meta("a", counterfactual=cf(0, True)), meta("b", counterfactual=cf(1, True))]
    assert any("baselines" in h for h in validate_dataset(two_base).hard)
    mixed = [meta("a", counterfactual=cf(0, True)), meta("b", counterfactual=cf(1, False, "mass"))]
    assert any("mixed variables" in h for h in validate_dataset(mixed).hard)
    with pytest.raises(ValueError):
        meta("c", role="test_ood_magnitude")              # OOD role needs ood_category


def test_nested_group_subsets():
    groups = [f"g{i}" for i in range(40)]
    sub = nested_group_subsets(groups, [0.1, 0.25, 0.5, 1.0], seed=0)
    assert sub[0.1] <= sub[0.25] <= sub[0.5] <= sub[1.0] == set(groups)
    assert len(sub[0.25]) == 10


# -- integrity: per-episode checksums ----------------------------------------------------------
def test_checksum_written_verified_and_detects_corruption(tmp_path):
    import h5py
    from physiworld_uav.data.writer import add_missing_checksums, episode_checksum, verify_shard
    p = tmp_path / "s.h5"
    a0, a1 = synthetic_episode(seed=0), synthetic_episode(seed=1)
    with ShardWriter(p) as w:
        w.write(a0, meta("ep0")); w.write(a1, meta("ep1"))
    assert verify_shard(p) == {"ep0": "ok", "ep1": "ok"}
    with h5py.File(p, "r") as f:
        stored = f["episodes/ep0"].attrs["checksum"]
    back, _ = read_episode(p, "ep0")
    assert stored == episode_checksum(back) and stored.startswith("sha256:")
    assert episode_checksum(back) != episode_checksum(read_episode(p, "ep1")[0])        # different data, different hash
    with h5py.File(p, "a") as f:                                                         # flip one value: silent corruption
        d = f["episodes/ep1/ground_truth/position_W"]
        x = d[()]; x[7, 1] += 1e-3; d[...] = x
    assert verify_shard(p) == {"ep0": "ok", "ep1": "mismatch"}


def test_legacy_episodes_without_checksum_are_reported_and_can_be_repaired(tmp_path):
    import h5py
    from physiworld_uav.data.writer import add_missing_checksums, verify_shard
    p = tmp_path / "s.h5"
    with ShardWriter(p) as w:
        w.write(synthetic_episode(), meta("ep0"))
    with h5py.File(p, "a") as f:
        del f["episodes/ep0"].attrs["checksum"]
    assert verify_shard(p) == {"ep0": "missing"}
    assert add_missing_checksums(p) == 1 and verify_shard(p) == {"ep0": "ok"} and add_missing_checksums(p) == 0


def test_resume_scan_removes_corrupted_episodes_and_validation_reports_them(tmp_path):
    import h5py
    from physiworld_uav.experiments import generate as G
    p = tmp_path / "shard_000.h5"
    with ShardWriter(p) as w:
        w.write(synthetic_episode(seed=0), meta("ep0")); w.write(synthetic_episode(seed=1), meta("ep1"))
    with h5py.File(p, "a") as f:
        d = f["episodes/ep1/ground_truth/velocity_W"]
        x = d[()]; x[3, 0] += 1.0; d[...] = x
    report = G.validate_dataset_dir(tmp_path)
    assert not report.ok and any("ep1" in h and "mismatch" in h for h in report.hard)
    rows, where = G._scan_shards(tmp_path, {})                       # what --resume / merge does
    assert [r["episode_id"] for r in rows] == ["ep0"] and "ep1" not in where       # ep1 will be generated again
    assert G.verify_shard(p) == {"ep0": "ok"}


# -- loader benchmark ---------------------------------------------------------------------------
def test_loader_benchmark_reports_rates_and_headroom(tmp_path):
    import json
    from physiworld_uav.data.loader_benchmark import benchmark, training_rate
    from test_models import synthetic_episode as sim_episode
    eps = [sim_episode(f"e{i}", i) for i in range(3)]
    run = tmp_path / "run"; run.mkdir()
    (run / "metrics.jsonl").write_text("\n".join(json.dumps({"epoch": i, "windows_per_s": r}) for i, r in enumerate((900.0, 1100.0, 1000.0))))
    assert training_rate(run) == 1000.0 and training_rate(None) is None and training_rate(tmp_path / "none") is None
    r = benchmark(tmp_path, episodes=eps, workers=(0, 2), batch_size=32, n_batches=5, noise="nominal", run_dir=run, horizon=10)
    assert r["n_episodes"] == 3 and r["n_windows"] > 100 and [x["num_workers"] for x in r["loader"]] == [0, 2]
    assert all(x["windows_per_s"] > 0 and x["windows"] > 0 for x in r["loader"])
    assert r["headroom_best_loader_over_training"] == pytest.approx(max(x["windows_per_s"] for x in r["loader"]) / 1000.0)


def test_imu_transport_status_and_warning_text():
    from types import SimpleNamespace as NS
    from physiworld_uav.data.dataset_builder import IMU_RAW_WARNING, imu_transport_status
    mk = lambda flag: NS(meta=NS(sim={} if flag is None else {"imu_transport_correction": flag}))  # noqa: E731
    assert imu_transport_status([]) == "unknown"
    assert imu_transport_status([mk(True), mk(True)]) == "corrected"
    assert imu_transport_status([mk(None), mk(False)]) == "raw"                  # old manifests have no flag: raw
    assert imu_transport_status([mk(True), mk(False)]) == "mixed"
    assert "RAW" in IMU_RAW_WARNING and "v0.6.0" in IMU_RAW_WARNING


# ------------------------------------------------------------------ model rate read from the dataset (20 Hz support)
def test_dataset_sim_reads_decimation_and_model_dt_from_the_manifest(tmp_path):
    import json
    from physiworld_uav.data.dataset_builder import dataset_decimation, dataset_sim
    assert dataset_sim(tmp_path) == {"control_decimation": 4, "model_dt": 0.02}          # no manifest: the 50 Hz defaults
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"sim": {"control_decimation": 10, "model_dt": 0.05, "physics_dt": 0.005}}) + "\n")
    assert dataset_sim(tmp_path) == {"control_decimation": 10, "model_dt": 0.05} and dataset_decimation(tmp_path) == 10
    (tmp_path / "manifest.jsonl").write_text("{}\n")
    assert dataset_decimation(tmp_path) == 4                                               # incomplete row: defaults


def test_evaluation_refuses_a_model_trained_at_another_rate(tmp_path):
    import json
    import pytest
    import torch
    from physiworld_uav.evaluation.run import load_model
    from physiworld_uav.models.world_model import ModelConfig, build_model, fit_stats
    from physiworld_uav.training import checkpoint as ck
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"sim": {"control_decimation": 10, "model_dt": 0.05}}) + "\n")
    name, m, _ = load_model("b1", tmp_path, torch.device("cpu"))
    assert abs(m.dt - 0.05) < 1e-12                                                        # B1 integrates at the dataset's model step
    from test_models import synthetic_episode as sim_episode
    eps = [sim_episode("a", 1)]
    mod = build_model("b2", stats=fit_stats(eps), cfg=ModelConfig(latent=8, embed=4, hidden=8), dt=0.02)
    ck.save_atomic({"model": mod.state_dict(), "kind": "b2", "stats": {k: getattr(mod, k).tolist() for k in ("imu_mu", "imu_sd", "act_mu", "act_sd", "v_mu", "v_sd", "w_mu", "w_sd", "p_sd", "dp_sd", "dv_sd", "dth_sd", "dw_sd")},
                     "model_cfg": {"latent": 8, "embed": 4, "hidden": 8}, "dt": 0.02}, tmp_path / "m.pt")
    with pytest.raises(SystemExit):
        load_model(f"b2={tmp_path / 'm.pt'}", tmp_path, torch.device("cpu"))
