"""Run 4 physics validation inside Isaac Sim (plan §5, §6.1, §7 TS5).

Every check is an isolated experiment with an analytic expectation; the
measured numbers are written to docs/models/validation_run4.json.
Run:  python -m pytest --isaac tests/isaac/test_physics_validation.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.isaac
ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "docs/models/validation_run4.json"


@pytest.fixture(scope="module")
def R(tmp_path_factory):
    out = tmp_path_factory.mktemp("probe") / "probe.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run([sys.executable, str(ROOT / "tests/isaac/physics_probe.py"), str(out)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    if not out.exists():
        pytest.fail(proc.stdout[-3000:] + proc.stderr[-3000:])
    res = json.loads(out.read_text())
    if "error" in res:
        pytest.fail(res["error"])
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(res, indent=1))
    return res


def test_pipeline_settings_applied(R):
    s = R["settings"]
    assert s["pipeline"] == "gpu_dynamics_cpu_readback" and s["gpu_dynamics"] and not s["suppress_readback"]
    assert s["enhanced_determinism"]
    assert np.allclose(s["gravity_W"], [0, 0, -9.81], atol=1e-5)
    assert all(v == [0.0, 0.0] for v in s["rigid_body_damping"].values())          # D6


def test_hover_equilibrium_and_imu(R):
    assert np.abs(R["hover"]["a_W"]).max() < 1e-4
    assert R["hover"]["imu_accel_mean"][2] == pytest.approx(9.81, abs=1e-2)


def test_free_fall_and_no_hidden_damping(R):
    assert R["free_fall"]["a_z"] == pytest.approx(-9.81, abs=1e-2)
    assert np.abs(R["free_fall"]["imu_accel_mean"]).max() < 1e-2                   # specific force 0
    assert R["coast"]["v_end"] == pytest.approx(R["coast"]["v0"], abs=1e-4)


def test_injected_force(R):
    r = R["inject_1N"]
    assert r["a_y"] == pytest.approx(r["expected"], rel=1e-2)
    assert r["imu_y"] == pytest.approx(r["expected"], rel=5e-3)


def test_motor_first_order_lag(R):
    assert R["motor_step"]["max_abs_err"] < 1e-6


@pytest.mark.parametrize("axis", ["yaw_torque", "roll_torque", "pitch_torque"])
def test_torque_response_matches_effective_inertia(R, axis):
    assert R[axis]["ang_acc"] == pytest.approx(R[axis]["expected"], rel=1e-2)


def test_gyro_equals_body_rate(R):
    assert np.allclose(R["gyro_vs_omegaB"]["omega_B"], R["gyro_vs_omegaB"]["gyro"], atol=1e-5)


def test_domain_binding(R):
    assert R["domain_binding"]["check_issues"] == []
    assert np.abs(R["domain_binding"]["hover_a_W"]).max() < 1e-3                    # true-mass trim of new domain


def test_wind_induced_acceleration_matches_model(R):
    assert R["wind_accel"]["a_x"] == pytest.approx(R["wind_accel"]["expected"], rel=1e-2)


def test_within_process_repeatability(R):
    assert R["determinism_max_abs_diff"] < 1e-4
