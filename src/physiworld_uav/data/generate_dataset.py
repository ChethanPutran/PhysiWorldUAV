

"""
Two classes for the world-model dataset generation pipeline:

    ControllerLibrary — all controller types from the config
    WindLibrary       — all wind types from the config

Both take the resolved Hydra config as input and are independent of the
Isaac Sim / OmniDrones environment. The generation script parses the drone
state into a plain dict, feeds it to these classes, and applies the outputs
to the drone.

Usage (inside the generation script's inner loop):

    ctrl_lib = ControllerLibrary(cfg, drone, device)
    wind_lib = WindLibrary(cfg, device)

    for phase_idx, phase in enumerate(scenario["phases"]):
        phase_key = f"{traj_id}:{phase_idx}"
        phase_dur = phase["duration_s"]
        for step in range(int(phase_dur / dt)):
            t = step * dt
            phase_elapsed = t
            state_dict = parse_drone_state(drone)

            wind_force = wind_lib.compute(
                phase["wind"], t, dt, phase_elapsed, phase_dur, phase_key)
            _apply_wind_to_prim(base_link_prim, wind_force)

            target_rate, target_thrust_N = ctrl_lib.compute(
                phase["controller"], state_dict, step, phase_elapsed, dt, phase_key)
            action = rate_controller(
                state_tensor,
                target_rate=torch.tensor([target_rate], device=device),
                target_thrust=torch.tensor([[target_thrust_N]], device=device),
            )
            drone.apply_action(action)
            sim.step(render=True)


Below is the full generation script. It imports the two classes from flight_controllers.py and 
handles: scenario allocation with coverage guarantees, paired-scenario co-generation, 
counterfactual replay, physics OOD, initial-condition sampling, event logging, quaternion continuity, 
and dataset verification.


Complete dataset generation driver for the world-model training set.

Reads `config/dataset_generation.yaml` (via Hydra), iterates every
scenario in every mode, and produces one .npz + .json per trajectory.

Handles:
  - Scenario allocation with coverage guarantees
  - Paired-scenario co-generation (shared params)
  - Counterfactual replay (variant 0 + replays under scaled wind)
  - Physics OOD (mass / inertia / drag modifiers)
  - Initial-condition sampling (pose, velocity, angular velocity)
  - Event manifest (maneuver/wind/gust timestamps)
  - Quaternion continuity enforcement
  - Post-hoc verification (identifiability + decorrelation)

Usage:
    python generate_dataset.py
    python generate_dataset.py experiment.output_dir=/datasets/wm_train
    python generate_dataset.py ood.wind_scale=2.0 \\
        experiment.output_dir=/datasets/wm_ood_magnitude
    python generate_dataset.py dataset.total_trajectories=5 \\
        experiment.output_dir=/tmp/smoke_test

"""
import os
import signal
import sys
import json
import time
from collections import defaultdict
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch


from .control_library import ControllerLibrary
from .wind_library import WindLibrary
from .utils import _enforce_quaternion_continuity, _euler_to_quat, _sample, _to_list, _unit,_set_physx_force
from pxr import PhysxSchema, Gf

MODES = ("hover", "translation", "acceleration", "yaw", "attitude", "combined")

# ═══════════════════════════════════════════════════════════════════
# 0.  Ctrl+C handler  (Kit's built-in handler segfaults mid-step)
# ═══════════════════════════════════════════════════════════════════
_STOP = [False]

def _sigint(signum, frame):
    _STOP[0] = True
    print("\n[signal] Ctrl+C — stopping after current trajectory...", flush=True)


# ═══════════════════════════════════════════════════════════════════
# 2.  Physics modifiers  (mass, inertia, drag)
# ═══════════════════════════════════════════════════════════════════

def read_base_drone_params(drone_yaml_path: Path, drone=None) -> dict:
    """
    Read base physical parameters for OOD modification.

    Handles two YAML layouts seen in OmniDrones assets:

      inertia: [0.007, 0.007, 0.012]                    # list (some drones)
      inertia: {xx: 0.007, xy: 0.0, xz: 0.0,
                yy: 0.007, yz: 0.0, zz: 0.012}          # dict (Hummingbird)

    Returns a dict with:
      mass              — the "nominal" mass for OOD scaling.
                          When `drone` is provided and its `MASS_0` differs
                          from the YAML `mass` (as it does for Hummingbird:
                          YAML 0.716, live 0.680), the LIVE value is used,
                          so `mass_scale=1.15` gives "15% heavier than what
                          the physics engine is actually using".
      yaml_mass         — the value from the YAML (kept for reference)
      live_mass         — the value from drone.MASS_0 (None if not provided)
      inertia           — [xx, yy, zz] diagonal, in kg·m²
      inertia_off_diag  — {xy, xz, yz} off-diagonal terms (usually 0)
      drag_coef         — the drone's aerodynamic drag coefficient
      arm_length        — arm length `l` (meters)
      rotor_config      — full rotor configuration dict (max RPM, force/moment
                          constants, rotor angles); returned so future OOD
                          experiments that vary rotor parameters have access
    """
    import yaml

    with open(drone_yaml_path) as f:
        y = yaml.safe_load(f) or {}

    # ── Mass ────────────────────────────────────────────────
    yaml_mass = float(y.get("mass", 0.716))
    live_mass = None
    if drone is not None:
        try:
            m = drone.MASS_0
            if hasattr(m, "item"):
                live_mass = float(m.item())
            else:
                live_mass = float(np.asarray(m).reshape(-1)[0])
        except Exception:
            live_mass = None
    mass = live_mass if live_mass is not None else yaml_mass

    # ── Inertia: list OR dict ───────────────────────────────
    inertia_raw = y.get("inertia", None)
    if inertia_raw is None:
        inertia_raw = y.get("diagonal_inertia", [0.007, 0.007, 0.012])

    if isinstance(inertia_raw, dict):
        inertia = [
            float(inertia_raw.get("xx", 0.007)),
            float(inertia_raw.get("yy", 0.007)),
            float(inertia_raw.get("zz", 0.012)),
        ]
        inertia_off_diag = {
            "xy": float(inertia_raw.get("xy", 0.0)),
            "xz": float(inertia_raw.get("xz", 0.0)),
            "yz": float(inertia_raw.get("yz", 0.0)),
        }
    else:
        inertia_list = _to_list(inertia_raw)
        inertia = [float(x) for x in inertia_list[:3]]
        while len(inertia) < 3:
            inertia.append(0.007)
        inertia_off_diag = {"xy": 0.0, "xz": 0.0, "yz": 0.0}

    # ── Drag coefficient and arm length ─────────────────────
    drag_coef = float(y.get("drag_coef", 0.2))
    arm_length = float(y.get("l", 0.34))

    # ── Rotor configuration (for future rotor-parameter OOD) ─
    rotor_cfg_raw = y.get("rotor_configuration", {}) or {}
    rotor_config = {
        "num_rotors": int(rotor_cfg_raw.get("num_rotors", 4)),
        "arm_lengths": [float(x) for x in _to_list(rotor_cfg_raw.get("arm_lengths", [0.17] * 4))],
        "directions": [float(x) for x in _to_list(rotor_cfg_raw.get("directions", [-1, 1, -1, 1]))],
        "force_constants": [float(x) for x in _to_list(rotor_cfg_raw.get("force_constants", [8.54858e-06] * 4))],
        "moment_constants": [float(x) for x in _to_list(rotor_cfg_raw.get("moment_constants", [1.36777e-07] * 4))],
        "max_rotation_velocities": [float(x) for x in _to_list(rotor_cfg_raw.get("max_rotation_velocities", [838.0] * 4))],
        "rotor_angles": [float(x) for x in _to_list(rotor_cfg_raw.get("rotor_angles", [0, 1.5708, 3.1416, -1.5708]))],
    }

    return {
        "mass": mass,
        "yaml_mass": yaml_mass,
        "live_mass": live_mass,
        "inertia": inertia,
        "inertia_off_diag": inertia_off_diag,
        "drag_coef": drag_coef,
        "arm_length": arm_length,
        "rotor_config": rotor_config,
    }


def resolve_physics_modifiers(spec_mods: dict, rng) -> dict:
    """Sample per-trajectory values from ranges."""
    out = {}
    for key in ("mass_scale", "inertia_scale", "drag_scale"):
        v = spec_mods.get(key, 1.0) if spec_mods else 1.0
        out[key] = _sample(rng, v)
    return out


def apply_physics_modifiers(base_link_prim, base_params: dict, resolved_mods: dict, drone=None):
    """
    Apply OOD mass / inertia / drag modifiers.

    Mass and inertia are written to the PhysX rigid-body API on the
    drone's base_link. Drag is handled in two phases:

      1. If `drone` is provided and exposes a `drag_coef` (or similar)
         attribute, scale it in place. This changes the actual aero force
         computation used by the drone model — the cleanest form of
         drag OOD, because the controller's thrust-to-command mapping is
         unaffected (the controller still thinks it's flying a nominal
         drone, which is the whole point of OOD).

      2. Otherwise, fall back to scaling the PhysX linear/angular damping
         attributes as a coarse proxy. This is less physically accurate
         but does change the dynamics of the rigid body.

    Returns a metadata dict with the applied values, written into the
    trajectory's JSON.
    """
    from pxr import PhysxSchema, Gf

    rb = PhysxSchema.PhysxRigidBodyAPI(base_link_prim)

    # ── Mass ────────────────────────────────────────────────
    new_mass = float(base_params["mass"]) * float(resolved_mods["mass_scale"])
    rb.GetMassAttr().Set(new_mass)

    # ── Inertia (diagonal only; off-diagonals are zero for Hummingbird) ─
    inertia_base = base_params["inertia"]
    inertia_scale = float(resolved_mods["inertia_scale"])
    new_inertia = [float(inertia_base[i]) * inertia_scale for i in range(3)]
    rb.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*new_inertia))

    # ── Drag: try drone attribute first, then USD damping ──
    drag_scale = float(resolved_mods["drag_scale"])
    drag_applied = {"mode": "none", "value": None}

    if drag_scale != 1.0:
        if drone is not None:
            # Try common attribute names for the aero drag coefficient
            for attr in ("drag_coef", "DRAG_COEF", "_drag_coef", "drag_coefficient"):
                if hasattr(drone, attr):
                    try:
                        old = float(getattr(drone, attr))
                        setattr(drone, attr, old * drag_scale)
                        drag_applied = {
                            "mode": "drone_attr", "attr": attr,
                            "old": old, "new": old * drag_scale,
                        }
                        break
                    except (AttributeError, TypeError, ValueError):
                        continue
            # Also scale per-rotor damping if that's how drag is modelled
            if drag_applied["mode"] == "none":
                if hasattr(drone, "rotor_configuration") and hasattr(drone.rotor_configuration, "drag_coef"):
                    try:
                        old = float(drone.rotor_configuration.drag_coef)
                        drone.rotor_configuration.drag_coef = old * drag_scale
                        drag_applied = {
                            "mode": "rotor_cfg", "old": old,
                            "new": old * drag_scale,
                        }
                    except Exception:
                        pass

        if drag_applied["mode"] == "none":
            # Fallback: USD damping
            base_damp = 0.1
            rb.GetLinearDampingAttr().Set(float(base_damp * drag_scale))
            rb.GetAngularDampingAttr().Set(float(base_damp * drag_scale))
            drag_applied = {
                "mode": "usd_damping_proxy",
                "linear_damping": base_damp * drag_scale,
                "angular_damping": base_damp * drag_scale,
            }

    return {
        "mass_kg": new_mass,
        "inertia_kgm2": new_inertia,
        "drag": drag_applied,
        "base_values": {
            "mass": float(base_params["mass"]),
            "inertia": [float(x) for x in base_params["inertia"]],
            "drag_coef": float(base_params.get("drag_coef", 0.0)),
        },
        "scales": dict(resolved_mods),
    }

# ═══════════════════════════════════════════════════════════════════
# 3.  Initial conditions  (pose + velocity per trajectory)
# ═══════════════════════════════════════════════════════════════════

def resolve_initial_conditions(mode_ic: dict, scenario_ic: dict, rng) -> dict:
    """
    Merge mode-level and scenario-level initial_conditions; resolve any
    ranges into concrete values.
    Scenario-level values override mode-level values. Returns a dict with:
    - pos_x, pos_y, pos_z_offset: position values
    - vel_xyz: velocity vector
    - roll, pitch, yaw: attitude angles
    """
    ic = {}
    for src in (mode_ic or {}, scenario_ic or {}):
        for k, v in src.items():
            ic[k] = v

    resolved = {}

    # Position
    pos_xy = ic.get("position_xy_range_m", [-2.0, 2.0])
    xy = _to_list(pos_xy)
    resolved["pos_x"] = rng.uniform(float(xy[0]), float(xy[1]))
    resolved["pos_y"] = rng.uniform(float(xy[0]), float(xy[1]))
    z_off = ic.get("position_z_offset_m", [-0.1, 0.1])
    zo = _to_list(z_off)
    resolved["pos_z_offset"] = rng.uniform(float(zo[0]), float(zo[1]))

    # Velocity
    v_range = _to_list(ic.get("velocity_magnitude_range_mps", [0.0, 0.0]))
    v_mag = rng.uniform(float(v_range[0]), float(v_range[1]))
    v_dir = ic.get("velocity_direction", "random_unit_vector")
    if v_dir == "random_unit_vector" or v_mag < 1e-6:
        v_vec = np.zeros(3, dtype=np.float32)
        if v_mag > 1e-6:
            raw = rng.normal(0, 1, size=3).astype(np.float32)
            v_vec = _unit(raw) * v_mag
    else:
        v_vec = _unit(np.asarray(v_dir, dtype=np.float32)) * v_mag
    resolved["vel_xyz"] = v_vec

    # Attitude
    yaw_range = _to_list(ic.get("yaw_range_rad", [-3.14159, 3.14159]))
    resolved["yaw"] = rng.uniform(float(yaw_range[0]), float(yaw_range[1]))

    tilt_mode = ic.get("tilt_mode_from_list", None)
    if tilt_mode is not None:
        tilt_modes = _to_list(tilt_mode)
        chosen = tilt_modes[rng.integers(len(tilt_modes))]
        mag_deg = _sample(rng, ic.get("tilt_magnitude_deg", [10.0, 25.0]))
        sign = rng.choice([-1, 1], size=2)
        roll_deg = (sign[0] * mag_deg) if chosen in ("roll_only", "diagonal") else 0.0
        pitch_deg = (sign[1] * mag_deg) if chosen in ("pitch_only", "diagonal") else 0.0
    else:
        roll_deg = _sample(rng, ic.get("roll_offset_deg", 0.0))
        pitch_deg = _sample(rng, ic.get("pitch_offset_deg", 0.0))

    resolved["roll"] = np.radians(roll_deg)
    resolved["pitch"] = np.radians(pitch_deg)

    # Angular velocity
    w_range = _to_list(ic.get("angular_velocity_magnitude_range_radps", [0.0, 0.0]))
    w_mag = rng.uniform(float(w_range[0]), float(w_range[1]))
    w_axis = ic.get("angular_velocity_axis", "random_unit_vector")
    if w_axis == "random_unit_vector" or w_mag < 1e-6:
        if w_mag < 1e-6:
            w_vec = np.zeros(3, dtype=np.float32)
        else:
            raw = rng.normal(0, 1, size=3).astype(np.float32)
            w_vec = _unit(raw) * w_mag
    else:
        w_vec = _unit(np.asarray(w_axis, dtype=np.float32)) * w_mag
    resolved["ang_vel_xyz"] = w_vec

    return resolved


def apply_initial_conditions(drone, mode_altitudes, ic_resolved, rng):
    """
    Set the drone's pose + velocity at trajectory start.
    Uses drone.set_world_poses / drone.set_velocities (from demo scripts).
    Falls back to _reset_idx if the setters are unavailable.
    """
    alt = float(mode_altitudes[rng.integers(len(mode_altitudes))])
    pos = np.array([
        ic_resolved["pos_x"],
        ic_resolved["pos_y"],
        alt + ic_resolved["pos_z_offset"],
    ], dtype=np.float32)

    quat = _euler_to_quat(
        ic_resolved["roll"], ic_resolved["pitch"], ic_resolved["yaw"]
    )

    vel6 = np.concatenate([
        ic_resolved["vel_xyz"],
        ic_resolved["ang_vel_xyz"],
    ]).astype(np.float32)

    try:
        drone.set_world_poses(
            torch.from_numpy(pos).unsqueeze(0).to(drone.device),
            torch.from_numpy(quat).unsqueeze(0).to(drone.device),
        )
        drone.set_velocities(torch.from_numpy(vel6).unsqueeze(0).to(drone.device))
    except Exception:
        # Fallback: just reset to spawn
        drone._reset_idx(torch.tensor([0], device=drone.device))

    return alt, pos, quat



# ═══════════════════════════════════════════════════════════════════
# 4.  Event extraction  (build manifest from phase spec)
# ═══════════════════════════════════════════════════════════════════

def build_event_manifest(phases: list, physics_mods: dict) -> list:
    """
    Pre-compute the event list for one trajectory by walking the phase spec.
    Events are (t_abs, type, id, params).
    """
    events = []
    t_abs = 0.0

    if physics_mods and any(
        v != 1.0 for k, v in physics_mods.items() if k.endswith("_scale")
    ):
        events.append({
            "t": 0.0, "type": "physics_perturbation", "id": "modifiers_applied",
            "params": physics_mods,
        })

    for phase_idx, phase in enumerate(phases):
        phase_dur = float(phase["duration_s"])
        ctrl = phase["controller"]
        ctrl_list = ctrl if isinstance(ctrl, list) else [ctrl]

        # maneuver_start for every controller in the phase
        for c in ctrl_list:
            ctype = c.get("type", "unknown")
            events.append({
                "t": t_abs, "type": "maneuver_start", "id": f"{ctype}_{phase_idx}",
                "params": {k: v for k, v in c.items() if k != "type"},
            })

        # wind events for this phase
        wind_spec = phase["wind"]
        wtype = wind_spec.get("type", "none")
        if wtype in ("constant", "sinusoidal", "dryden", "rotating", "random_multi"):
            events.append({
                "t": t_abs, "type": "wind_change", "id": f"{wtype}_{phase_idx}",
                "params": dict(wind_spec),
            })
        elif wtype == "step":
            step_t = float(wind_spec.get("step_time_s", 0.0))
            if step_t < phase_dur:
                events.append({
                    "t": t_abs + step_t, "type": "wind_change",
                    "id": f"step_{phase_idx}",
                    "params": {
                        "from_N": wind_spec.get("from_magnitude_N"),
                        "to_N": wind_spec.get("to_magnitude_N"),
                    },
                })
        elif wtype == "multi_step":
            for s in wind_spec.get("steps", []):
                events.append({
                    "t": t_abs + float(s["t_s"]), "type": "wind_change",
                    "id": f"multi_step_{phase_idx}",
                    "params": {"magnitude_N": s["magnitude_N"]},
                })
        elif wtype == "gust":
            g_t = float(wind_spec.get("gust_time_s", 0.0))
            g_d = float(wind_spec.get("duration_s", 0.5))
            events.append({
                "t": t_abs + g_t, "type": "gust_start", "id": f"gust_{phase_idx}",
                "params": {"peak_N": wind_spec.get("peak_magnitude_N")},
            })
            events.append({
                "t": t_abs + g_t + g_d, "type": "gust_end", "id": f"gust_{phase_idx}",
            })
        elif wtype == "gust_mid":
            # trigger_at_progress is a range; the actual trigger is sampled
            # per trajectory by WindLibrary, so we approximate with the midpoint
            prog = _to_list(wind_spec.get("trigger_at_progress", [0.5, 0.5]))
            mid = 0.5 * (float(prog[0]) + float(prog[1]))
            dur = _to_list(wind_spec.get("duration_s", [0.5, 0.5]))
            mid_d = 0.5 * (float(dur[0]) + float(dur[1]))
            events.append({
                "t": t_abs + mid * phase_dur, "type": "gust_start",
                "id": f"gust_mid_{phase_idx}",
            })
            events.append({
                "t": t_abs + mid * phase_dur + mid_d,
                "type": "gust_end", "id": f"gust_mid_{phase_idx}",
            })

        # maneuver_end for every controller in the phase
        for c in ctrl_list:
            ctype = c.get("type", "unknown")
            events.append({
                "t": t_abs + phase_dur, "type": "maneuver_end",
                "id": f"{ctype}_{phase_idx}",
            })

        t_abs += phase_dur

    events.sort(key=lambda e: e["t"])
    return events


# ═══════════════════════════════════════════════════════════════════
# 5.  Trajectory save  (quaternion continuity + npz + json)
# ═══════════════════════════════════════════════════════════════════
def save_trajectory(
    traj: dict, meta: dict, out_dir: Path, traj_idx: int, suffix: str = ""
):
    out_dir.mkdir(parents=True, exist_ok=True)

    traj["orientation"] = _enforce_quaternion_continuity(
        np.asarray(traj["orientation"], dtype=np.float32)
    )

    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in traj.items() if len(v)}
    stem = f"trajectory_{traj_idx:04d}{suffix}"
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, **arrays)

    meta["num_steps"] = len(traj["t"])
    meta["timestamp"] = time.time()
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(meta, indent=2, default=str))

    return npz_path, json_path


# ── Trajectory helpers ─────────────────────────────────────────────
def _init_trajectory():
    return {
        "t": [],
        "position": [],
        "orientation": [],
        "velocity": [],
        "angular_velocity": [],
        "accel_imu": [],
        "gyro_imu": [],
        "action": [],
        "wind_force_N": [],
        "mass": [],
        "inertia": [],
        "drag_coeff": [],
        "thrust_N": [],
        "mode": [],
        "wind_type": [],
        "wind_magnitude_N": [],
        "wind_direction": [],
        "pair_id": [],
        "seed": [],
    }

# ═══════════════════════════════════════════════════════════════════
# 6.  Single-variant runner
# ═══════════════════════════════════════════════════════════════════

def run_variant(
    cfg,
    drone,
    rate_controller,
    sim,
    base_link_prim,
    ctrl_lib: ControllerLibrary,
    wind_lib: WindLibrary,
    phases: list,
    traj_seed: int,
    wind_multiplier: float = 1.0,
    action_replay: list = None,
    collect_actions: bool = False,
    log_frequency: int = 100,
):
    """
    Run one trajectory variant. If `action_replay` is provided, bypass
    controllers and use those actions. If `collect_actions` is True,
    return the list of actions used (for later replay).
    """
    dt = float(cfg.sim.dt)
    rng = np.random.default_rng(traj_seed)

    # Reset drone and phase state
    drone._reset_idx(torch.tensor([0], device=drone.device))
    sim.render()

    # Reset controller and wind state (for phase-dependent controllers)
    ctrl_lib.clear_phase_state()
    wind_lib.clear_wind_state()

    traj = _init_trajectory()

    action_buffer = []
    step_counter = 0
    global_t = 0.0

    for phase_idx, phase in enumerate(phases):
        phase_dur = float(phase["duration_s"])
        n_steps = int(round(phase_dur / dt))
        phase_key = f"{traj_seed}:{phase_idx}"

        for k in range(n_steps):
            phase_elapsed = k * dt

            wind_type = phase["wind"].get("type", "none")

            # ── Wind ────────────────────────────────────────────
            w_force = wind_lib.compute(
                wind_type, global_t, dt, phase_elapsed, phase_dur, phase_key
            )
            if wind_multiplier != 1.0:
                w_force = w_force * wind_multiplier
            if action_replay is None or step_counter == 0:
                _set_physx_force(base_link_prim, w_force)
            else:
                # For replay, wind IS different from the buffer's wind
                _set_physx_force(base_link_prim, w_force)

            # ── State read ──────────────────────────────────────
            state_np = (
                drone.get_state()[..., :13]
                .reshape(-1, 13)[0].detach().cpu().numpy()
            )
            state_dict = {
                "position": state_np[0:3],
                "orientation": state_np[3:7],
                "velocity": state_np[7:10],
                "angular_velocity": state_np[10:13],
            }

            # ── Action ──────────────────────────────────────────
            if action_replay is not None:
                act_flat = np.asarray(action_replay[step_counter], dtype=np.float32)
                # act_flat is [rate_r, rate_p, rate_y, thrust_N]
                target_rate = act_flat[:3]
                target_thrust_N = float(act_flat[3])
            else:
                target_rate, target_thrust_N = ctrl_lib.compute(
                    phase["controller"], state_dict, k, phase_elapsed, dt, phase_key
                )
                if collect_actions:
                    action_buffer.append(np.concatenate([
                        target_rate, [target_thrust_N]
                    ]).astype(np.float32))

            # ── Apply to RateController + step ──────────────────
            action = rate_controller(
                drone.get_state()[..., :13].squeeze(0),
                target_rate=torch.tensor(
                    [target_rate], device=cfg.sim.device, dtype=torch.float32
                ),
                target_thrust=torch.tensor(
                    [[target_thrust_N]], device=cfg.sim.device, dtype=torch.float32
                ),
            )
            drone.apply_action(action)
            sim.step(render=True)

            # ── Log ─────────────────────────────────────────────
            state_np_post = (
                drone.get_state()[..., :13]
                .reshape(-1, 13)[0].detach().cpu().numpy()
            )
            traj["t"].append(global_t)
            traj["position"].append(state_np_post[0:3].tolist())
            traj["orientation"].append(state_np_post[3:7].tolist())
            traj["velocity"].append(state_np_post[7:10].tolist())
            traj["angular_velocity"].append(state_np_post[10:13].tolist())
            traj["accel_imu"].append(drone.get_accel_imu()[0].detach().cpu().numpy().tolist())
            traj["gyro_imu"].append(drone.get_gyro_imu()[0].detach().cpu().numpy().tolist())
            traj["action"].append(
                np.concatenate([target_rate, [target_thrust_N]]).astype(np.float32).tolist()
            )
            traj["mass"].append(float(drone.MASS_0))
            traj["inertia"].append(_to_list(float(drone.INERTIA_0)))
            traj["drag_coeff"].append(float(getattr(drone, "drag_coef", 0.0)))
            traj["thrust_N"].append(float(target_thrust_N))
            traj["wind_type"].append(wind_type)
            traj["wind_magnitude_N"].append(float(np.linalg.norm(w_force)))
            traj["wind_direction"].append(_to_list(w_force / np.linalg.norm(w_force) if np.linalg.norm(w_force) > 0 else [0, 0, 0]))
            traj["wind_force_N"].append(w_force.tolist())
            traj["pair_id"].append(phase.get("pair_id", None))
            traj["seed"].append(traj_seed)

            step_counter += 1
            global_t += dt

            if cfg.experiment.verbose and step_counter % log_frequency == 0:
                print(
                    f"  [step {step_counter}] t={global_t:6.2f}s  "
                    f"p=({state_np_post[0]:+.2f},{state_np_post[1]:+.2f},"
                    f"{state_np_post[2]:+.2f})  "
                    f"v=({state_np_post[7]:+.2f},{state_np_post[8]:+.2f},"
                    f"{state_np_post[9]:+.2f})  "
                    f"wind={np.linalg.norm(w_force):.2f}N",
                    flush=True,
                )

            if _STOP[0]:
                print("[stop] breaking trajectory early")
                return traj, action_buffer

            # Ground-hit safeguard
            if state_np_post[2] < 0.05:
                print(f"  [safeguard] ground hit at t={global_t:.2f}s")
                return traj, action_buffer

    return traj, action_buffer


# ═══════════════════════════════════════════════════════════════════
# 7.  Scenario allocation  (weight + coverage guarantee)
# ═══════════════════════════════════════════════════════════════════

def allocate_scenarios(cfg) -> list:
    """
    Return a flat list of (mode_name, scenario_dict, instance_idx) tuples
    sized to hit the configured trajectory count per mode.
    """
    total = int(cfg.dataset.total_trajectories)
    out = []

    for mode_name in MODES:
        if mode_name not in cfg.dataset.modes:
            continue
        mode_cfg = cfg.dataset.modes[mode_name]
        mode_count = max(1, int(round(total * float(mode_cfg.fraction))))
        scenarios = list(mode_cfg.definition.scenarios)
        weights = [float(s.weight) for s in scenarios]
        w_sum = sum(weights)

        if bool(mode_cfg.definition.sampling.guarantee_coverage):
            counts = [1] * len(scenarios)
            remaining = mode_count - len(scenarios)
            while remaining > 0:
                # Pick scenario with highest weight-per-assigned-slot
                ratios = [weights[i] / (counts[i] + 1) for i in range(len(scenarios))]
                idx = int(np.argmax(ratios))
                counts[idx] += 1
                remaining -= 1
        else:
            counts = [max(0, int(round(mode_count * w / w_sum))) for w in weights]
            # Fix rounding drift
            while sum(counts) < mode_count:
                counts[int(np.argmax(weights))] += 1
            while sum(counts) > mode_count:
                counts[int(np.argmin(counts))] -= 1

        for scenario, n in zip(scenarios, counts):
            for k in range(n):
                out.append((mode_name, scenario, k))

    return out


def presample_pairs(cfg, allocation: list) -> dict:
    """
    For paired scenarios, pre-sample shared parameters once per pair instance.
    Returns {pair_prefix: [params_per_instance, ...]}.
    """
    pair_map = {}  # (mode, scenario_name) -> (pair_name, prefix)
    pair_defs = {}  # pair_name -> list of member names
    pair_shared = {}  # pair_name -> list of shared keys

    for mode_name in MODES:
        if mode_name not in cfg.dataset.modes:
            continue
        defn = cfg.dataset.modes[mode_name].definition
        if "paired_scenarios" not in defn:
            continue
        for p in defn.paired_scenarios:
            pname = p.name
            members = list(p.members)
            shared = list(p.shared) if "shared" in p else []
            prefix = p.get("pair_id_prefix", pname)
            pair_defs[pname] = members
            pair_shared[pname] = shared
            for m in members:
                pair_map[(mode_name, m)] = (pname, prefix)

    # Count instances per pair = min over member counts
    member_counts = defaultdict(int)
    for mode_name, scenario, _ in allocation:
        key = (mode_name, scenario.name)
        member_counts[key] += 1

    pair_instances = {}
    for pname, members in pair_defs.items():
        # Find member scenario's mode
        m_mode = None
        for (mn, sn), _ in pair_map.items():
            if sn in members:
                m_mode = mn
                break
        if m_mode is None:
            continue
        counts = [member_counts[(m_mode, m)] for m in members]
        n_instances = min(counts)
        pair_instances[pname] = {
            "prefix": pair_map[(m_mode, members[0])][1],
            "n_instances": n_instances,
            "shared": pair_shared[pname],
            "params": [],
            "consumed": 0,
        }

    # Pre-sample shared params per pair instance
    rng = np.random.default_rng(int(cfg.experiment.seed) + 999)
    for pname, info in pair_instances.items():
        for _ in range(info["n_instances"]):
            info["params"].append(_sample_shared(rng, info["shared"]))

    return pair_instances, pair_map



def _extract_scenario_field(scenario, field_name):
    """
    Field path lookup by convention:
      - "target_change_rad", "rate_class", "ramp_fraction", ... → phases[0].controller
      - "wind_magnitude_N", "wind_direction"                    → phases[0].wind
      - "total_duration_s", "abs_target_change_rad"             → computed/derived
    """
    # Direct fields on the scenario
    if field_name in scenario:
        return scenario[field_name]

    # Controller field in the first phase
    ctrl = scenario["phases"][0]["controller"]
    ctrl_list = ctrl if isinstance(ctrl, list) else [ctrl]
    for c in ctrl_list:
        if field_name in c:
            return c[field_name]

    # Wind field in the first phase
    wind = scenario["phases"][0].get("wind", {})
    wind_key = field_name.replace("wind_", "")   # "wind_magnitude_N" → "magnitude_N"
    if wind_key in wind:
        return wind[wind_key]
    if field_name in wind:
        return wind[field_name]

    # Derived fields
    if field_name == "total_duration_s":
        return sum(float(p["duration_s"]) for p in scenario["phases"])
    if field_name == "abs_target_change_rad":
        val = _extract_scenario_field(scenario, "target_change_rad")
        return abs(val) if not isinstance(val, list) else [abs(v) for v in val]

    raise KeyError(f"Can't extract shared field {field_name!r} from {scenario['name']}")



def _sample_shared(rng, shared_keys, member_scenarios):
    """Extract the range of each shared field from the scenario specs,
    assert they agree, then sample once."""
    out = {}
    for k in shared_keys:
        values = [_extract_scenario_field(s, k) for s in member_scenarios]
        # All members must declare the same range for this shared field
        first = values[0]
        for v in values[1:]:
            if v != first:
                raise ValueError(
                    f"Shared field {k!r} differs across pair members: "
                    f"{values}. It must be identical."
                )
        out[k] = _sample(rng, first)
    return out



def apply_pair_params(scenario, shared_params, rng):
    """
    Return a modified scenario spec where the shared fields have been
    replaced by the pre-sampled values. Non-shared fields retain their
    ranges and are sampled independently per trajectory.
    """
    import copy
    out = copy.deepcopy(scenario)

    for field_name, value in shared_params.items():
        # Try direct write on scenario
        if field_name in out:
            out[field_name] = value
            continue

        # Controller fields — write into every controller in the first phase
        written = False
        for phase in out["phases"]:
            ctrl = phase["controller"]
            ctrl_list = ctrl if isinstance(ctrl, list) else [ctrl]
            for c in ctrl_list:
                if field_name in c:
                    c[field_name] = value
                    written = True

        # Wind fields
        wind = out["phases"][0].get("wind", {})
        wind_key = field_name.replace("wind_", "")
        if wind_key in wind:
            wind[wind_key] = value
            written = True
        elif field_name in wind:
            wind[field_name] = value
            written = True

        if not written:
            raise KeyError(f"Could not inject shared field {field_name!r}")

    return out

def _extract_from_meta_or_npz(meta: dict, npz, field_name: str):
    """
    Extract a field by name from either metadata (JSON) or trajectory (NPZ).

    Resolution order:
      1. Top-level metadata field
      2. Physics modifier block (meta["physics"][<key>])
      3. Events array (maneuver_start params)
      4. Computed trajectory statistics (peak_velocity, peak_accel, etc.)
    """
    import numpy as np

    # 1. Top-level metadata
    if field_name in meta:
        return meta[field_name]

    # 2. Physics modifiers
    if field_name.startswith("physics_"):
        key = field_name[len("physics_"):]
        if "physics" in meta and key in meta["physics"]:
            return meta["physics"][key]
    if field_name in ("mass_scale", "inertia_scale", "drag_scale"):
        scales = meta.get("physics", {}).get("scales", {})
        if field_name in scales:
            return scales[field_name]

    # 3. Event params (controller configuration lives here)
    for ev in meta.get("events", []):
        if ev["type"] == "maneuver_start" and field_name in ev.get("params", {}):
            return ev["params"][field_name]

    # 4. Computed trajectory statistics
    if field_name == "peak_velocity":
        return float(np.max(np.linalg.norm(npz["velocity"], axis=1)))
    if field_name == "peak_angular_velocity":
        return float(np.max(np.linalg.norm(npz["angular_velocity"], axis=1)))
    if field_name == "peak_acceleration":
        dt = float(npz["t"][1] - npz["t"][0]) if len(npz["t"]) > 1 else 0.01
        accel = np.diff(npz["velocity"], axis=0) / dt
        return float(np.max(np.linalg.norm(accel, axis=1)))
    if field_name == "peak_yaw_change":
        q = npz["orientation"]
        yaws = np.unwrap(2 * np.arctan2(q[:, 3], q[:, 0]))
        return float(yaws.max() - yaws.min())
    if field_name == "peak_roll" or field_name == "peak_pitch":
        q = npz["orientation"]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        if field_name == "peak_roll":
            roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        else:
            sinp = np.clip(2 * (w * y - z * x), -1, 1)
            roll = np.arcsin(sinp)
        return float(np.max(np.abs(roll)))

    raise KeyError(
        f"Can't extract field {field_name!r} from metadata or trajectory. "
        f"Available metadata keys: {list(meta.keys())}"
    )



def verify_pair(
    pair_id: str,
    member_a: dict,          # {"npz": <Path>, "json": <Path>}
    member_b: dict,          # {"npz": <Path>, "json": <Path>}
    differs: list[str],
    thresholds: dict = None,
    strict: bool = True,
) -> dict:
    """
    Verify that a generated pair satisfies the causal-pair contract.

    Checks
    ------
    1. pair_id matches between the two members' metadata
    2. The two members have different scenario names
    3. Both members have the same number of timesteps
    4. Actions are identical (up to numerical noise)
    5. Wind forces are identical (up to numerical noise)
    6. State trajectories DIFFER by more than the threshold
    7. Every field in `differs` has different values across members

    Parameters
    ----------
    pair_id : str
        The pair id, as recorded in metadata (e.g. "heading_180_rate_0003").
    member_a, member_b : dict
        Two entries from the manifest, each with "npz" and "json" keys.
    differs : list[str]
        Field names that are EXPECTED to differ (e.g. ["rate_class"]).
    thresholds : dict, optional
        Keys: min_position_divergence_m, min_orientation_divergence,
              min_velocity_divergence_mps, min_angular_velocity_divergence_radps,
              action_atol, wind_atol.
        Defaults shown in the code below.
    strict : bool
        If True, raises AssertionError on failure. If False, records failure
        in the return dict and continues.

    Returns
    -------
    report : dict
        Diagnostic summary. Contains `pass`: bool, `failure_reason`: str|None,
        and the divergence metrics.
    """
    import json
    import numpy as np
    from pathlib import Path

    t = {
        "min_position_divergence_m":              0.02,
        "min_orientation_divergence":             0.005,
        "min_velocity_divergence_mps":            0.02,
        "min_angular_velocity_divergence_radps":  0.02,
        "action_atol":                            1e-3,
        "wind_atol":                              1e-2,
        "step_count_atol":                        0,
    }
    if thresholds:
        t.update(thresholds)

    report = {
        "pair_id": pair_id,
        "scenario_a": None, "scenario_b": None,
        "num_steps": None,
        "action_max_diff": None,
        "wind_max_diff": None,
        "position_divergence_m": None,
        "orientation_divergence": None,
        "velocity_divergence_mps": None,
        "angular_velocity_divergence_radps": None,
        "differs_values": {},
        "pass": False,
        "failure_reason": None,
    }

    def _fail(reason):
        report["failure_reason"] = reason
        if strict:
            raise AssertionError(f"[{pair_id}] {reason}")
        return report

    # ── Load ──────────────────────────────────────────────────────
    a_npz = np.load(member_a["npz"])
    b_npz = np.load(member_b["npz"])
    a_meta = json.loads(Path(member_a["json"]).read_text())
    b_meta = json.loads(Path(member_b["json"]).read_text())

    report["scenario_a"] = a_meta.get("scenario")
    report["scenario_b"] = b_meta.get("scenario")

    # ── Check 1: pair_id match ────────────────────────────────────
    if a_meta.get("pair_id") != pair_id:
        return _fail(f"member_a pair_id {a_meta.get('pair_id')!r} != {pair_id!r}")
    if b_meta.get("pair_id") != pair_id:
        return _fail(f"member_b pair_id {b_meta.get('pair_id')!r} != {pair_id!r}")

    # ── Check 2: different scenarios ──────────────────────────────
    if a_meta.get("scenario") == b_meta.get("scenario"):
        return _fail(
            f"Both members are scenario {a_meta.get('scenario')!r} — "
            f"pair is degenerate by construction"
        )

    # ── Check 3: same step count ──────────────────────────────────
    n_a = int(len(a_npz["action"]))
    n_b = int(len(b_npz["action"]))
    report["num_steps"] = n_a
    if abs(n_a - n_b) > t["step_count_atol"]:
        return _fail(
            f"Step counts differ: a={n_a}, b={n_b}. "
            f"One member may have hit the ground early. "
            f"Truncate to min and regenerate, or loosen the ground-hit safeguard."
        )

    # ── Check 4: identical commands ───────────────────────────────
    action_diff = float(np.max(np.abs(a_npz["action"] - b_npz["action"])))
    report["action_max_diff"] = action_diff
    if action_diff > t["action_atol"]:
        return _fail(
            f"Actions differ by {action_diff:.4f} (atol={t['action_atol']}). "
            f"Pair is not controlled — the shared params were not injected. "
            f"Check that _sample_shared returns real values, not None."
        )

    # ── Check 5: identical wind ──────────────────────────────────
    wind_diff = float(np.max(np.abs(a_npz["wind_force_N"] - b_npz["wind_force_N"])))
    report["wind_max_diff"] = wind_diff
    if wind_diff > t["wind_atol"]:
        return _fail(
            f"Wind differs by {wind_diff:.4f} N (atol={t['wind_atol']}). "
            f"Most likely cause: shared field 'wind_magnitude_N' was not actually "
            f"shared between members (the _sample_shared stub bug)."
        )

    # ── Check 6: state must diverge ───────────────────────────────
    pos_div = float(np.mean(
        np.linalg.norm(a_npz["position"] - b_npz["position"], axis=1)
    ))
    qa = a_npz["orientation"]
    qb = b_npz["orientation"]
    # Sign-invariant quaternion distance
    ori_div = float(np.mean(1.0 - np.abs(np.sum(qa * qb, axis=1))))
    vel_div = float(np.mean(
        np.linalg.norm(a_npz["velocity"] - b_npz["velocity"], axis=1)
    ))
    av_div = float(np.mean(
        np.linalg.norm(a_npz["angular_velocity"] - b_npz["angular_velocity"], axis=1)
    ))

    report["position_divergence_m"] = pos_div
    report["orientation_divergence"] = ori_div
    report["velocity_divergence_mps"] = vel_div
    report["angular_velocity_divergence_radps"] = av_div

    state_differs = (
        pos_div > t["min_position_divergence_m"]
        or ori_div > t["min_orientation_divergence"]
        or vel_div > t["min_velocity_divergence_mps"]
        or av_div > t["min_angular_velocity_divergence_radps"]
    )
    if not state_differs:
        return _fail(
            f"Pair is degenerate — state does not diverge. "
            f"pos={pos_div:.4f}m, ori={ori_div:.4f}, vel={vel_div:.4f}m/s, "
            f"ang_vel={av_div:.4f}rad/s. "
            f"All below thresholds. Widen the range of the differing parameter "
            f"(e.g. mass_scale 1.15→1.25, rate_class slow→fast) or extend the "
            f"trajectory duration."
        )

    # ── Check 7: differs fields actually differ ───────────────────
    for field in differs:
        try:
            va = _extract_from_meta_or_npz(a_meta, a_npz, field)
            vb = _extract_from_meta_or_npz(b_meta, b_npz, field)
        except KeyError as e:
            return _fail(f"Could not extract differs field {field!r}: {e}")
        report["differs_values"][field] = (va, vb)
        if va == vb:
            return _fail(
                f"differs field {field!r} is identical across members "
                f"(value={va!r}). The pair does not vary the declared dimension."
            )

    report["pass"] = True
    return report
    

def verify_all_pairs(
    out_dir: Path,
    cfg=None,
    thresholds: dict = None,
    fail_on_error: bool = False,
) -> dict:
    """
    Walk every pair in `out_dir`, verify each, print a summary table.

    Returns
    -------
    summary : dict
        {"n_pairs": int, "n_passed": int, "n_failed": int,
         "reports": [verify_pair_report, ...],
         "failure_reasons": [str, ...]}
    """
    import json
    from collections import defaultdict

    # Group trajectories by pair_id
    by_pair = defaultdict(list)
    for json_path in sorted(out_dir.glob("trajectory_*.json")):
        meta = json.loads(json_path.read_text())
        pid = meta.get("pair_id")
        if not pid:
            continue
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        by_pair[pid].append({"npz": npz_path, "json": json_path})

    # Determine `differs` per pair instance from the config
    differs_by_pair = {}
    if cfg is not None:
        for mode_name in ("translation", "yaw", "attitude", "combined", "hover"):
            if mode_name not in cfg.dataset.modes:
                continue
            defn = cfg.dataset.modes[mode_name].definition
            if "paired_scenarios" not in defn:
                continue
            for p in defn.paired_scenarios:
                prefix = p.get("pair_id_prefix", p.name)
                differs_by_pair[prefix] = list(p.get("differs", []))

    # Verify each pair
    reports = []
    n_passed = 0
    n_failed = 0
    failure_reasons = []

    print("\n" + "=" * 78)
    print(" Pair verification")
    print("=" * 78)
    print(f"{'pair_id':38s}  {'status':8s}  {'Δpos (m)':>9s}  {'Δori':>7s}  {'Δvel':>7s}")
    print("-" * 78)

    for pid, members in sorted(by_pair.items()):
        if len(members) != 2:
            print(f"{pid:38s}  {'SKIP':8s}  ({len(members)} members, expected 2)")
            continue

        # Infer pair prefix from pair_id: "physics_nominal_vs_perturbed_0003" → "physics_nominal_vs_perturbed"
        prefix = "_".join(pid.split("_")[:-1])
        differs = differs_by_pair.get(prefix, [])

        try:
            report = verify_pair(
                pid, members[0], members[1], differs,
                thresholds=thresholds, strict=True,
            )
            reports.append(report)
            n_passed += 1
            print(
                f"{pid:38s}  {'PASS':8s}  "
                f"{report['position_divergence_m']:>9.4f}  "
                f"{report['orientation_divergence']:>7.4f}  "
                f"{report['velocity_divergence_mps']:>7.4f}"
            )
        except AssertionError as e:
            n_failed += 1
            failure_reasons.append(str(e))
            # Reload partial metrics for display
            try:
                r = verify_pair(
                    pid, members[0], members[1], differs,
                    thresholds=thresholds, strict=False,
                )
                reports.append(r)
                print(
                    f"{pid:38s}  {'FAIL':8s}  "
                    f"{r['position_divergence_m'] or 0.0:>9.4f}  "
                    f"{r['orientation_divergence'] or 0.0:>7.4f}  "
                    f"{r['velocity_divergence_mps'] or 0.0:>7.4f}"
                )
            except Exception:
                print(f"{pid:38s}  {'FAIL':8s}  (could not extract metrics)")

    print("-" * 78)
    print(f"  {n_passed} passed, {n_failed} failed, {n_passed + n_failed} pairs total")

    if failure_reasons:
        print("\nFailure reasons:")
        for reason in failure_reasons[:10]:   # first 10 to keep output manageable
            print(f"  • {reason}")
        if len(failure_reasons) > 10:
            print(f"  … and {len(failure_reasons) - 10} more")

    if fail_on_error and n_failed > 0:
        raise RuntimeError(
            f"{n_failed} pair verification(s) failed. See output above."
        )

    return {
        "n_pairs": n_passed + n_failed,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "reports": reports,
        "failure_reasons": failure_reasons,
    }


# ═══════════════════════════════════════════════════════════════════
# 8.  Trajectory generation  (single + counterfactual)
# ═══════════════════════════════════════════════════════════════════

def generate_trajectory(
    cfg, drone, rate_controller, sim, base_link_prim,
    base_params, ctrl_lib, wind_lib,
    mode_name, scenario, instance_idx,
    pair_info: dict, out_dir: Path, traj_idx: int, seed_base: int,
):
    """Generate one trajectory (or one counterfactual pair group)."""
    traj_seed = seed_base + hash((mode_name, scenario.name, instance_idx)) % 1_000_000
    rng = np.random.default_rng(traj_seed)

    mode_cfg = cfg.dataset.modes[mode_name].definition

    # ── Initial conditions ──────────────────────────────────
    ic_resolved = resolve_initial_conditions(
        mode_cfg.get("initial_conditions", {}),
        scenario.get("initial_conditions", {}),
        rng,
    )
    alt, spawn_pos, spawn_quat = apply_initial_conditions(
        drone, mode_cfg.get("altitudes_m", [1.5]), ic_resolved, rng
    )

    # ── Physics modifiers ───────────────────────────────────
    physics_mods = resolve_physics_modifiers(
        scenario.get("physics_modifiers", {}), rng
    )
    physics_applied = apply_physics_modifiers(
        base_link_prim, base_params, physics_mods, drone=drone
    )
    sim.reset()
    drone._reset_idx(torch.tensor([0], device=drone.device))

    # ── Build events + trajectory metadata ─────────────────
    phases = scenario.phases
    events = build_event_manifest(phases, physics_applied)
    total_dur = sum(float(p["duration_s"]) for p in phases)

    # ── Counterfactual or single ────────────────────────────
    is_cf = bool(scenario.get("counterfactual", cfg.dataset.counterfactual.enabled
                              if "counterfactual" in cfg.dataset else False))

    pair_id = None
    if pair_info is not None and instance_idx < pair_info["n_instances"]:
        pair_id = f"{pair_info['prefix']}_{instance_idx:04d}"

    base_meta = {
        "experiment": str(cfg.experiment.name),
        "seed": int(cfg.experiment.seed),
        "mode": mode_name,
        "scenario": scenario.name,
        "instance": instance_idx,
        "pair_id": pair_id,
        "duration_s": total_dur,
        "spawn_position": spawn_pos.tolist(),
        "spawn_orientation": spawn_quat.tolist(),
        "initial_conditions": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in ic_resolved.items()
        },
        "physics": physics_applied,
        "events": events,
        "wind_config": OmegaConf.to_container(cfg.wind, resolve=True)
        if "wind" in cfg else {},
        "ood": OmegaConf.to_container(cfg.ood, resolve=True) if "ood" in cfg else {},
    }

    saved_paths = []

    if is_cf:
        multipliers = _to_list(cfg.dataset.counterfactual.wind_multipliers)
        if len(multipliers) < 2:
            multipliers = [1.0, 2.0]

        # Variant 0: nominal, collect actions
        print(f"  [variant 0] wind × {multipliers[0]}")
        traj0, actions = run_variant(
            cfg, drone, rate_controller, sim, base_link_prim,
            ctrl_lib, wind_lib, phases, traj_seed,
            wind_multiplier=float(multipliers[0]),
            collect_actions=True,
        )
        meta0 = dict(base_meta, variant=0, wind_multiplier=float(multipliers[0]))
        paths = save_trajectory(traj0, meta0, out_dir, traj_idx, suffix="_v0")
        saved_paths.append(paths)

        # Variants 1..N: replay actions with scaled wind
        for i, mult in enumerate(multipliers[1:], start=1):
            print(f"  [variant {i}] wind × {mult}")
            traj_i, _ = run_variant(
                cfg, drone, rate_controller, sim, base_link_prim,
                ctrl_lib, wind_lib, phases, traj_seed,
                wind_multiplier=float(mult),
                action_replay=actions,
            )
            meta_i = dict(base_meta, variant=i, wind_multiplier=float(mult))
            paths = save_trajectory(traj_i, meta_i, out_dir, traj_idx, suffix=f"_v{i}")
            saved_paths.append(paths)

    else:
        print(f"  [single] {mode_name}/{scenario.name}")
        traj, _ = run_variant(
            cfg, drone, rate_controller, sim, base_link_prim,
            ctrl_lib, wind_lib, phases, traj_seed,
        )
        paths = save_trajectory(traj, base_meta, out_dir, traj_idx)
        saved_paths.append(paths)

    return saved_paths


def _find_drone_prim_path(drone):
    """Locate the drone's root prim by name in the stage."""
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    name = getattr(drone, "name", "Hummingbird")
    candidates = [
        p.GetPath().pathString for p in stage.Traverse()
        if name in p.GetPath().pathString and "/World/envs/env_0" in p.GetPath().pathString
    ]
    if candidates:
        candidates.sort(key=len)
        return candidates[0]
    raise RuntimeError(f"Could not find drone prim matching {name!r}")

# ═══════════════════════════════════════════════════════════════════
# 9.  Main  (setup → generate → verify)
# ═══════════════════════════════════════════════════════════════════

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="dataset_generation",
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones import init_simulation_app
    sim_app = init_simulation_app(cfg)
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    try:
        # ── Imports AFTER app is up ─────────────────────────────
        import omni_drones.utils.scene as scene_utils
        from omni.isaac.core.simulation_context import SimulationContext
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni_drones.robots.drone import MultirotorBase
        from omni_drones.sensors.camera import Camera, PinholeCameraCfg
        from .control_library import ControllerLibrary
        from .wind_library import WindLibrary


        device = cfg.sim.device

        # ── Setup sim + drone ───────────────────────────────────
        sim = SimulationContext(
            stage_units_in_meters=cfg.sim.stage_units_in_meters,
            physics_dt=cfg.sim.dt,
            rendering_dt=cfg.sim.dt,
            sim_params=cfg.sim,
            backend=cfg.sim.backend,
            device=device,
        )

        # ── Drone ────────────────────────────────────────────────
        drone, controller = MultirotorBase.make(
            cfg.drone_model.name,
            cfg.drone_model.controller,
            device,
        )
    
        spawn = torch.zeros(1, 3, device=device)
        spawn[:, 2] = 1.5


        drone.spawn(translations=spawn)
        scene_utils.design_scene()


        # ── Camera ───────────────────────────────────────────────
        camera_sensor = None
        if cfg.camera.enabled:
            cam_cfg = PinholeCameraCfg(
                sensor_tick=0,
                resolution=tuple(int(v) for v in cfg.camera.resolution),
                data_types=["rgb"],
            )
            camera_sensor = Camera(cam_cfg)
            cam_prim = f"/World/envs/env_0/{drone.name}_0/base_link/Camera"
            camera_sensor.spawn([cam_prim])
            print(f"[camera] Spawned at {cam_prim}")

        sim.reset()

        if cfg.camera.enabled:
            camera_sensor.initialize(f"/World/envs/env_0/{drone.name}_*/base_link/Camera") # type: ignore
            print("[camera] Initialized")
        
        drone.initialize()
        sim.play()
        sim.render()

        # ── Locate base_link and read base params ───────────────
        drone_root = _find_drone_prim_path(drone)
        base_link_prim = get_prim_at_path(f"{drone_root}/base_link")
        if base_link_prim is None or not base_link_prim.IsValid():
            raise RuntimeError(f"base_link not found at {drone_root}/base_link")

        drone_yaml = Path(
            f"/workspace/PhysiWorldUAV/OmniDrones/omni_drones/robots/assets/usd/"
            f"{cfg.drone_model.name.lower()}.yaml"
        )

        base_params = read_base_drone_params(drone_yaml, drone=drone)
        print(f"[drone] YAML mass = {base_params['yaml_mass']:.4f} kg")
        print(f"[drone] Live MASS_0 = {base_params['live_mass']} kg "
            f"(used as OOD nominal)" if base_params['live_mass'] is not None else "")
        print(f"[drone] Inertia = {base_params['inertia']} kg·m²")
        print(f"[drone] drag_coef = {base_params['drag_coef']:.4f}")

        # ── Controller and wind libraries ───────────────────────
        ctrl_lib = ControllerLibrary(cfg, drone, device=device,
                                     seed=int(cfg.experiment.seed))
        wind_lib = WindLibrary(cfg, device=device,
                               seed=int(cfg.experiment.seed))

        # ── Output directory ────────────────────────────────────
        out_dir = Path(cfg.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Scenario allocation + pair sampling ─────────────────
        allocation = allocate_scenarios(cfg)
        pair_instances, pair_map = presample_pairs(cfg, allocation)

        n_total = len(allocation)
        print(f"\n[plan] {n_total} trajectory slots across {len(allocation)//1} slots")
        for mode_name in ["hover", "translation", "acceleration", "yaw", "attitude", "combined"]:
            if mode_name in cfg.dataset.modes:
                count = sum(1 for m, _, _ in allocation if m == mode_name)
                print(f"  {mode_name:12s}: {count}")

        # ── Generation loop ─────────────────────────────────────
        t_start = time.time()
        saved = []
        for slot_idx, (mode_name, scenario, instance_idx) in enumerate(allocation):
            if _STOP[0]:
                break
            print(f"\n[traj {slot_idx + 1}/{n_total}] {mode_name}/{scenario.name} #{instance_idx}")

            # Determine pair info for this scenario
            pair_info = None
            for pname, info in pair_instances.items():
                if any(sn == scenario.name for (_, sn) in pair_map.keys()
                       if pair_map[(_, sn)][0] == pname):
                    pair_info = info
                    break

            try:
                paths = generate_trajectory(
                    cfg, drone, controller, sim, base_link_prim,
                    base_params, ctrl_lib, wind_lib,
                    mode_name, scenario, instance_idx,
                    pair_info, out_dir, slot_idx,
                    seed_base=int(cfg.experiment.seed) * 1000,
                )
                saved.extend(paths)
            except Exception as e:
                print(f"[ERROR] trajectory failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # ── Write manifest ──────────────────────────────────────
        manifest = {
            "experiment": str(cfg.experiment.name),
            "seed": int(cfg.experiment.seed),
            "total_trajectories_saved": len(saved),
            "trajectories": [
                {"npz": str(p[0].name), "json": str(p[1].name)} for p in saved
            ],
            "wall_time_s": time.time() - t_start,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
        print(f"\n[done] {len(saved)} trajectory files written to {out_dir}")

        # ── Verification pass ───────────────────────────────────
        _verify_outputs(out_dir, cfg)

        # Verify all pairs
        pair_summary = verify_all_pairs(
            out_dir,
            cfg=cfg,
            thresholds={
                "min_position_divergence_m": 0.02,
                "min_orientation_divergence": 0.005,
                "min_velocity_divergence_mps": 0.02,
            },
            fail_on_error=False,    # print failures but finish the run
        )

        # Optionally promote failures to a fatal error
        if pair_summary["n_failed"] > 0:
            print(f"\n[warn] {pair_summary['n_failed']} pair(s) failed verification — "
                f"they will be dropped from training.")
            # Save the failure list so the training script can exclude them
            (out_dir / "failed_pairs.json").write_text(
                json.dumps(pair_summary["failure_reasons"], indent=2)
            )

    except KeyboardInterrupt:
        print("\n[stop] KeyboardInterrupt", flush=True)
    finally:
        try:
            sim_app.close()
        except Exception:
            pass
        print("[done] Simulation closed.", flush=True)
        os._exit(0)

# ═══════════════════════════════════════════════════════════════════
# 10. Verification pass  (identifiability + decorrelation checks)
# ═══════════════════════════════════════════════════════════════════

def _verify_outputs(out_dir: Path, cfg):
    print("\n" + "=" * 60)
    print(" Verification")
    print("=" * 60)

    npz_files = sorted(out_dir.glob("trajectory_*.npz"))
    if not npz_files:
        print("[warn] no trajectory files to verify")
        return

    per_mode = defaultdict(lambda: {
        "count": 0, "quat_fail": 0,
        "corr_u_wind": [], "corr_v_wind": [],
    })

    for npz_path in npz_files:
        meta_path = npz_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        mode = meta.get("mode", "unknown")
        d = np.load(npz_path)

        q = d["orientation"]
        dots = np.sum(q[:-1] * q[1:], axis=1)
        n_flip = int(np.sum(dots < 0))
        if n_flip > 0:
            per_mode[mode]["quat_fail"] += 1

        u = d["action"]
        v = d["velocity"]
        F = d["wind_force_N"]
        F_mag = np.linalg.norm(F, axis=1)
        v_mag = np.linalg.norm(v, axis=1)
        u_thrust = u[:, 3]

        if F_mag.std() > 1e-6 and len(u_thrust) > 2:
            per_mode[mode]["corr_u_wind"].append(
                float(np.corrcoef(u_thrust, F_mag)[0, 1])
            )
            per_mode[mode]["corr_v_wind"].append(
                float(np.corrcoef(v_mag, F_mag)[0, 1])
            )
        per_mode[mode]["count"] += 1

    print(f"\n{'mode':14s} {'n':>4s} {'quat_fails':>11s} {'|corr(u_thrust,F)|':>20s} {'|corr(|v|,F)|':>16s}")
    print("-" * 72)
    for mode in ["hover", "translation", "acceleration", "yaw", "attitude", "combined"]:
        if mode not in per_mode:
            continue
        s = per_mode[mode]
        cu = np.mean(np.abs(s["corr_u_wind"])) if s["corr_u_wind"] else 0.0
        cv = np.mean(np.abs(s["corr_v_wind"])) if s["corr_v_wind"] else 0.0
        print(f"{mode:14s} {s['count']:>4d} {s['quat_fail']:>11d} "
              f"{cu:>20.3f} {cv:>16.3f}")

    print("\nThresholds:")
    print("  quat_fails == 0         — quaternions continuous")
    print("  |corr(u_thrust, |F|)| < 0.3   — action-wind decorrelation OK")
    print("  |corr(|v|, |F|)|      > 0.2   — wind affecting state OK")
    print("  (hover is expected to fail both corr checks; that is by design)")
    print("=" * 60)


if __name__ == "__main__":
    main()
