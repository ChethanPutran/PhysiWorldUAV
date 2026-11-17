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
