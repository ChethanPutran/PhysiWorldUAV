"""Dataset schema v2 — single source of truth (plan §6.5; requirement.md #4).

Sampling semantics at the physics rate (k = 0…N, t_k = k·dt):

* ``inst`` — sampled at the instant t_k (after step k−1 → k); N+1 rows.
* ``intv`` — applied/held over the interval [t_k, t_{k+1}); N rows.

Groups (ground truth / observations / actions are kept separate):

* ``time``          timestamps
* ``ground_truth``  simulator truth (state, motors, wrench components)
* ``observations``  sensor measurements (IMU) — model inputs (D7)
* ``actions``       actuator commands
* ``control``       references and tracking errors (closed loop only)
* ``environment``   wind (auxiliary; never a model input, D1/D2)

Model-ready layout (plan DS6): state = p_W(3) | v_W(3) | q_wxyz(4) | ω_B(3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

SCHEMA_VERSION = "2.0"

#: Canonical model state layout (the simulator layout p|q|v|ω_W stays inside core/).
MODEL_STATE_FIELDS: tuple[tuple[str, int], ...] = (
    ("ground_truth/position_W", 3),
    ("ground_truth/velocity_W", 3),
    ("ground_truth/orientation_wxyz", 4),
    ("ground_truth/angular_velocity_B", 3),
)
MODEL_STATE_LAYOUT = ("p_W[0:3]", "v_W[3:6]", "q_wxyz[6:10]", "omega_B[10:13]")
MODEL_OBS_FIELDS: tuple[tuple[str, int], ...] = (
    ("observations/imu_accel_B", 3),
    ("observations/imu_gyro_B", 3),
)

#: Data types (requirement.md #2) and split roles (plan DS8).
DATA_TYPES = ("closed_loop", "open_loop", "physics_validation")
ID_ROLES = ("train", "val_id", "test_id")
OOD_CATEGORIES = ("magnitude", "direction", "temporal", "gust", "physics", "trajectory", "joint")
SPLIT_ROLES = ID_ROLES + tuple(f"{p}_ood_{c}" for c in OOD_CATEGORIES for p in ("val", "test")) \
    + ("physics_validation",)

#: Keys that must never appear in a primary (model-input) sample.
AUXILIARY_PREFIXES = ("environment/", "ground_truth/wrench/", "control/")


@dataclass(frozen=True)
class FieldSpec:
    path: str
    width: tuple[int, ...]          # trailing shape; () for scalars per row
    dtype: str
    unit: str
    frame: str
    sem: str                        # inst | intv
    required: bool
    description: str


_F = FieldSpec
FIELDS: tuple[FieldSpec, ...] = (
    _F("time/t", (), "float64", "s", "-", "inst", True, "simulation time t_k = k·dt"),
    # ground truth state
    _F("ground_truth/position_W", (3,), "float32", "m", "world ENU", "inst", True, "base_link origin position"),
    _F("ground_truth/orientation_wxyz", (4,), "float32", "-", "body→world", "inst", True,
       "Hamilton unit quaternion, sign-continuous"),
    _F("ground_truth/velocity_W", (3,), "float32", "m/s", "world", "inst", True, "linear velocity"),
    _F("ground_truth/angular_velocity_B", (3,), "float32", "rad/s", "body FLU", "inst", True,
       "body angular velocity ω_B = R_WBᵀ ω_W (thesis state)"),
    _F("ground_truth/angular_velocity_W", (3,), "float32", "rad/s", "world", "inst", True,
       "world angular velocity (diagnostic; simulator native)"),
    _F("ground_truth/linear_acceleration_W", (3,), "float32", "m/s²", "world", "inst", False,
       "DERIVED: central difference of PhysX velocity"),
    _F("ground_truth/angular_acceleration_B", (3,), "float32", "rad/s²", "body", "inst", False,
       "DERIVED: central difference of ω_B"),
    _F("ground_truth/rotor_speed", (4,), "float32", "rad/s", "rotor", "inst", True,
       "Ω_i = throttle_i·Ω_max,i after the motor lag (D3 model action)"),
    _F("ground_truth/rotor_thrust", (4,), "float32", "N", "+Z_B", "inst", True, "k_F·Ω_i²"),
    # wrench applied by our force modules (populated from Run 4 on; zeros + flag before)
    _F("ground_truth/wrench/external_force_W", (3,), "float32", "N", "world", "intv", True,
       "total force injected by project force models (exactly what was applied)"),
    _F("ground_truth/wrench/external_torque_B", (3,), "float32", "N·m", "body", "intv", True,
       "total torque injected by project force models (about the base-link origin)"),
    # per-component wrench (only components enabled for the episode are present; plan F9)
    *[f for comp in ("rotor_drag", "body_drag", "ground_effect", "aero_torque", "program")
      for f in (_F(f"ground_truth/wrench/{comp}_force_W", (3,), "float32", "N", "world", "intv", False,
                   f"{comp} force applied over the interval"),
                _F(f"ground_truth/wrench/{comp}_torque_B", (3,), "float32", "N·m", "body", "intv", False,
                   f"{comp} torque about the base-link origin"))],
    # observations
    _F("observations/imu_accel_B", (3,), "float32", "m/s²", "body (sensor=body)", "inst", True,
       "specific force at the base link from Isaac IMUSensor, with the sensor's -ω_B×v_B transport term removed when "
       "sim.imu_transport_correction is true (datasets from v0.6.0; the Run 5 pilot v0.5.0 holds the raw sensor). Clean; noise applied at load time, D7"),
    _F("observations/imu_gyro_B", (3,), "float32", "rad/s", "body", "inst", True, "IMU angular rate"),
    _F("observations/imu_valid", (), "bool", "-", "-", "inst", True, "fresh sensor frame at t_k"),
    # actions
    _F("actions/rotor_cmd", (4,), "float32", "[-1,1]", "-", "intv", True,
       "applied (clamped) OmniDrones rotor command, held over the control interval"),
    _F("actions/rotor_cmd_raw", (4,), "float32", "-", "-", "intv", True, "controller output before clamping"),
    _F("actions/saturated", (4,), "bool", "-", "-", "intv", True, "rotor_cmd_raw outside [-1,1]"),
    _F("actions/rotor_speed_cmd", (4,), "float32", "rad/s", "rotor", "intv", True,
       "steady-state Ω target of the command: sqrt((cmd+1)/2)·Ω_max (Q6)"),
    _F("actions/collective_thrust_cmd", (), "float32", "N", "+Z_B", "intv", True,
       "commanded collective thrust (NaN when the mode has none)"),
    _F("actions/body_rate_cmd", (3,), "float32", "rad/s", "body", "intv", True,
       "commanded body rates (NaN when the mode has none)"),
    # control (closed loop)
    _F("control/reference/position_W", (3,), "float32", "m", "world", "intv", False, "reference position"),
    _F("control/reference/velocity_W", (3,), "float32", "m/s", "world", "intv", False, "reference velocity"),
    _F("control/reference/acceleration_W", (3,), "float32", "m/s²", "world", "intv", False, "reference accel"),
    _F("control/reference/yaw", (), "float32", "rad", "world", "intv", False, "reference yaw"),
    _F("control/reference/roll", (), "float32", "rad", "-", "intv", False, "commanded roll (attitude mode)"),
    _F("control/reference/pitch", (), "float32", "rad", "-", "intv", False, "commanded pitch (attitude mode)"),
    _F("control/tracking_error/position_W", (3,), "float32", "m", "world", "intv", False, "p − p_ref"),
    _F("control/tracking_error/velocity_W", (3,), "float32", "m/s", "world", "intv", False, "v − v_ref"),
    # environment (auxiliary)
    _F("environment/wind_W", (3,), "float32", "m/s", "world", "inst", True, "wind at the CoM at t_k"),
    _F("environment/wind_applied_W", (3,), "float32", "m/s", "world", "intv", True,
       "wind used to compute the wrench for interval k"),
    _F("environment/wind_rotor_W", (4, 3), "float32", "m/s", "world", "intv", False,
       "wind at each rotor hub for interval k (differs from the CoM wind only with a spatial gradient)"),
)
BY_PATH: dict[str, FieldSpec] = {f.path: f for f in FIELDS}


def expected_rows(spec: FieldSpec, n_steps: int) -> int:
    return n_steps + 1 if spec.sem == "inst" else n_steps


def validate_arrays(arrays: Mapping[str, np.ndarray], *, data_type: str) -> list[str]:
    """Schema check of one episode's physics-rate arrays. Returns a list of issues."""
    issues: list[str] = []
    if "time/t" not in arrays:
        return ["missing time/t"]
    n_steps = int(np.asarray(arrays["time/t"]).shape[0]) - 1
    if n_steps < 1:
        return [f"episode has {n_steps + 1} samples; need ≥ 2"]
    for spec in FIELDS:
        if spec.path not in arrays:
            closed_only = spec.path.startswith("control/")
            if spec.required and not (closed_only and data_type != "closed_loop"):
                issues.append(f"missing required field {spec.path}")
            continue
        arr = np.asarray(arrays[spec.path])
        want = (expected_rows(spec, n_steps),) + spec.width
        if arr.shape != want:
            issues.append(f"{spec.path}: shape {arr.shape} != {want}")
        if arr.dtype != np.dtype(spec.dtype):
            issues.append(f"{spec.path}: dtype {arr.dtype} != {spec.dtype}")
    unknown = set(arrays) - set(BY_PATH)
    if unknown:
        issues.append(f"unknown fields (not in schema): {sorted(unknown)}")
    return issues


def schema_table_markdown() -> str:
    rows = ["| Field | Shape | dtype | Unit | Frame | Sem. | Req. | Description |",
            "|---|---|---|---|---|---|---|---|"]
    for f in FIELDS:
        shape = "(N+1" if f.sem == "inst" else "(N"
        shape += "".join(f",{w}" for w in f.width) + ")"
        rows.append(f"| `{f.path}` | {shape} | {f.dtype} | {f.unit} | {f.frame} | {f.sem} | "
                    f"{'yes' if f.required else 'no'} | {f.description} |")
    return "\n".join(rows)
