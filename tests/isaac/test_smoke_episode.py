"""Run 3 exit criterion: a real Isaac episode in schema v2 with complete metadata.

Run inside the isaac-sim container:  pytest -m isaac --isaac tests/isaac
Isaac's SimulationApp.close() terminates the interpreter, so generation runs
in a subprocess and the test inspects its output.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from physiworld_uav.data.dataset_validator import validate_episode
from physiworld_uav.data.schema import validate_arrays
from physiworld_uav.data.writer import read_episode

pytestmark = pytest.mark.isaac
ROOT = Path(__file__).resolve().parents[2]


def _cli(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    return subprocess.run([sys.executable, "-m", "physiworld_uav.experiments.generate", *args],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory) -> Path:
    out_root = tmp_path_factory.mktemp("datasets")
    proc = _cli("run", "experiment=smoke", f"dataset.output_root={out_root}")
    out = out_root / "smoke_v0.3.0"
    if not (out / "index.json").exists():
        pytest.fail(f"generation produced no index.json\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return out


def _episode(out: Path):
    index = json.loads((out / "index.json").read_text())
    row = index["episodes"][0]
    return index, read_episode(out / row["shard"], row["episode_id"])


def test_outputs_and_provenance(smoke):
    for f in ("manifest.jsonl", "resolved_config.yaml", "provenance.json", "index.json", "validation.json"):
        assert (smoke / f).exists(), f
    prov = json.loads((smoke / "provenance.json").read_text())
    assert prov["config_hash"] and prov["project_git"]["commit"]
    assert prov["simulator"]["isaac_sim"].startswith("4.1")


def test_schema_and_metadata_complete(smoke):
    index, (arrays, meta) = _episode(smoke)
    assert validate_arrays(arrays, data_type=meta.data_type) == []
    assert meta.status == "completed" and meta.n_steps == 800
    assert meta.config_hash == json.loads((smoke / "provenance.json").read_text())["config_hash"]
    for k in ("episode", "initial_conditions", "trajectory", "wind", "excitation"):
        assert k in meta.seeds
    assert meta.controller["policy"] == "true" and meta.controller["mode"] == "position"
    assert meta.plan["episode_id"] == meta.episode_id


def test_hover_physics_sanity(smoke):
    _, (arrays, meta) = _episode(smoke)
    ic = np.asarray(meta.initial_conditions["position"])
    p = arrays["ground_truth/position_W"]
    assert np.allclose(p[0], ic, atol=1e-3)                          # reset to the planned IC
    assert np.abs(p - ic).max() < 0.05                                 # Lee hover holds
    omega = arrays["ground_truth/rotor_speed"]
    mass = meta.drone["mass_kg"]
    kf = np.asarray(meta.drone["force_constants"])
    assert (kf * omega[-1] ** 2).sum() == pytest.approx(mass * 9.81, rel=5e-3)   # true-mass trim
    q = arrays["ground_truth/orientation_wxyz"][0]
    assert 2 * np.degrees(np.arctan2(q[3], q[0])) == pytest.approx(meta.initial_conditions["yaw_deg"], abs=0.1)


def test_dataset_marked_complete(smoke):
    """Run 4: IMU prim created before sim.reset() + CPU readback (decision (a)) → valid IMU."""
    index, (arrays, meta) = _episode(smoke)
    assert validate_episode(arrays, meta).ok
    assert index["complete"]
    assert arrays["observations/imu_valid"].all()
    # IMU specific force equals R_WBᵀ((v_k − v_(k−1))/dt + g e₃) from the logged state
    from physiworld_uav.core.state_extractor import quat_rotate_inverse_np
    v = arrays["ground_truth/velocity_W"].astype(np.float64)
    q = arrays["ground_truth/orientation_wxyz"].astype(np.float64)
    a = np.diff(v, axis=0) / meta.sim["physics_dt"] + np.array([0.0, 0.0, 9.81])
    f_state = quat_rotate_inverse_np(q[1:], a)
    assert np.abs(arrays["observations/imu_accel_B"][1:] - f_state).max() < 1e-2
    assert meta.drone_realised["total_mass_kg"] == pytest.approx(0.716, rel=1e-6)
    assert meta.sim["pipeline"] == "gpu_dynamics_cpu_readback"


def test_regenerate_reproduces_episode(smoke):
    eid = json.loads((smoke / "index.json").read_text())["episodes"][0]["episode_id"]
    _cli("regenerate", "--dataset", str(smoke), "--episode", eid)
    diffs = json.loads((smoke / f"regenerate_{eid}.json").read_text())
    print("regenerate max |Δ| per field:", json.dumps(diffs, indent=1))
    assert diffs["ground_truth/position_W"] < 1e-4
    assert diffs["actions/rotor_cmd"] < 1e-4
