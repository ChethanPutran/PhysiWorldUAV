#!/usr/bin/env python3
"""
collect_data_wind_hover.py
================
Hover the drone in mid-air, log state + IMU + wind disturbance, plot each
trajectory, and visualize disturbance-force vectors and body forces in Isaac Sim.

The drone model in OmniDrones does not expose a hardware IMU, so the IMU
is DERIVED from the ground-truth state:

    accelerometer (body)  =  R^T @ (a_world - g_world)   ← specific force
    gyroscope     (body)  =  ω_body                      ← already body-frame

Where a_world is the finite-difference of world-frame velocity and
g_world = [0, 0, -9.81]. At perfect hover (a_world = 0, quat = identity),
the accelerometer reads [0, 0, +9.81] m/s², i.e. 1 g upward — as a real
IMU would.

The script attempts to attach an Isaac Sim physics IMU to the drone base link.
If that API is unavailable in this Isaac Sim/OmniDrones installation, it falls
back to the synthetic IMU computed from ground-truth state. The CSV/NPZ fields
remain the same. Debug-draw overlays show the configured disturbance force on
a grid and approximate thrust/gravity/disturbance force vectors at the drone.

Examples:
    python collect_data_wind_hover.py
    python collect_data_wind_hover.py collection.num_trajectories=1 control_mode=controlled wind.type=constant wind.constant.magnitude=2.0
    python collect_data_wind_hover.py collection.num_trajectories=1 control_mode=fixed_input wind.type=constant wind.constant.magnitude=2.0
    python collect_data_wind_hover.py wind.type=constant wind.constant.magnitude=2.0
    python collect_data_wind_hover.py wind.type=step wind.step.magnitude=4.0 wind.step.step_time=5.0
    python collect_data_wind_hover.py wind.type=sinusoidal control_mode=controlled wind.sinusoidal.amplitude=2.0
    python collect_data_wind_hover.py wind.type=gust wind.gust.magnitude=5.0 wind.gust.start_time=3.0
    python collect_data_wind_hover.py wind.type=dryden wind.dryden.sigma=0.8
    python collect_data_wind_hover.py collection.num_trajectories=1 collection.steps_per_trajectory=1000
    python collect_data_wind_hover.py collection.num_trajectories=1 wind.type=constant wind.constant.magnitude=2.0
"""

import csv
import json
import math
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import matplotlib.pyplot as plt

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# ── Optional IMU + debug visualization helpers ──────────────────────
def _create_imu_sensor(base_link_path, dt):
    """Try to create a physics-based IMU attached to the drone base link.

    Isaac Sim 4.x installations differ in sensor extension/API availability.
    Returns None if the real sensor cannot be created; caller then uses the
    synthetic state-derived IMU.
    """
    try:
        import omni.isaac.sensor
        from omni.isaac.sensor import IMUSensor

        imu_path = f"{base_link_path}/IMU"
        imu = IMUSensor(
            prim_path=imu_path,
            name="drone_imu",
            frequency=max(1, int(round(1.0 / float(dt)))),
            translation=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        if hasattr(imu, "initialize"):
            imu.initialize()
        print(f"[imu] Physics IMU created at {imu_path}")
        return imu
    except Exception as exc:
        print(
            "[imu] Could not initialize Isaac Sim IMUSensor; using "
            f"state-derived synthetic IMU instead ({type(exc).__name__}: {exc})"
        )
        return None

# ── Quaternion math ────────────────────────────────────────────────

def quaternion_to_matrix(q):
    """
    q: (..., 4), quaternion in wxyz order.
    Returns (..., 3, 3) body-to-world rotation matrix.
    Supports both NumPy arrays and PyTorch tensors.
    """
    if torch.is_tensor(q):
        q = torch.nn.functional.normalize(q, dim=-1)
        w, x, y, z = q.unbind(dim=-1)

        return torch.stack([
            1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w),
            2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
            2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y),
        ], dim=-1).reshape(q.shape[:-1] + (3, 3))

    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1e-12)

    w, x, y, z = np.moveaxis(q, -1, 0)

    R = np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w),
        2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y),
    ], axis=-1)

    return R.reshape(q.shape[:-1] + (3, 3))



def compute_distributed_drag(
    position_world,       # (3,), COM position
    velocity_world,       # (3,), COM velocity
    angular_velocity_world,  # (3,), rad/s
    R_world_body,         # (3, 3), body -> world
    point_offsets_body,   # (N, 3), offsets from COM
    point_CdA_xyz,        # (N, 3), allocated effective drag area
    base_wind_world,      # (3,), background + temporal wind
    spatial_cfg,
    rho=1.225,
):
    """
    Returns total aerodynamic force (N) and torque about COM (N m).

    point_CdA_xyz must partition the effective drag area across
    sample points; do not assign the full-body area to every point.
    """
    offsets_body = torch.as_tensor(
        point_offsets_body,
        device=position_world.device,
        dtype=position_world.dtype,
    )
    CdA = torch.as_tensor(
        point_CdA_xyz,
        device=position_world.device,
        dtype=position_world.dtype,
    )

    # Convert point offsets to world coordinates.
    offsets_world = offsets_body @ R_world_body.T
    points_world = position_world + offsets_world

    # Rigid-body velocity at each sample point.
    point_velocities_world = velocity_world + torch.linalg.cross(
        angular_velocity_world.expand_as(offsets_world),
        offsets_world,
        dim=-1,
    )

    # Use the SAME wind-field equation as the visualization.
    local_wind_world = _wind_velocity_at_positions(
        points_world,
        base_wind_world,
        spatial_cfg,
    )

    relative_velocity_world = (
        point_velocities_world - local_wind_world
    )

    # Convert relative velocity into body coordinates.
    relative_velocity_body = relative_velocity_world @ R_world_body

    # Approximate drag independently at each sample point.
    drag_body = (
        -0.5 * rho * CdA
        * relative_velocity_body.abs()
        * relative_velocity_body
    )

    drag_world = drag_body @ R_world_body.T

    total_force_world = drag_world.sum(dim=0)

    total_torque_world = torch.linalg.cross(
        offsets_world,
        drag_world,
        dim=-1,
    ).sum(dim=0)

    return total_force_world, total_torque_world


def aerodynamic_drag_world(
    velocity_world,
    wind_world,
    R_world_body,
    cfg,
):
    """
    velocity_world:   (3,) drone velocity in world frame, m/s
    wind_world:       (3,) air velocity in world frame, m/s
    R_world_body:     (3, 3) body-to-world rotation matrix
    Returns:
        drag force in world coordinates, N
    """

    rho = cfg.air_density

    # Drone velocity relative to the surrounding air.
    v_rel_world = velocity_world - wind_world

    # Transform relative airflow into the drone body frame.
    v_rel_body = R_world_body.T @ v_rel_world

    # Effective drag coefficients multiplied by reference areas (m^2).
    # These values must be calibrated or justified for your vehicle.
    CdA = torch.tensor(
        cfg.drag_coefficient_area_xyz,
        device=velocity_world.device,
        dtype=velocity_world.dtype,
    )

    # Component-wise quadratic drag.
    drag_body = (
        -0.5
        * rho
        * CdA
        * torch.abs(v_rel_body)
        * v_rel_body
    )

    # Transform the force back to world coordinates.
    drag_world = R_world_body @ drag_body

    return drag_world

def _read_imu_sensor(imu):
    """Return (accel_body, gyro_body) from an Isaac IMUSensor reading or None."""
    if imu is None:
        return None
    try:
        reading = imu.get_sensor_reading()
        if reading is None:
            return None

        # Isaac Sim releases expose these values through slightly different
        # reading types; support common attribute and dictionary forms.
        def get_field(*names):
            for name in names:
                if isinstance(reading, dict) and name in reading:
                    return reading[name]
                if hasattr(reading, name):
                    return getattr(reading, name)
            return None

        accel = get_field("lin_acc", "linear_acceleration", "acceleration")
        gyro = get_field("ang_vel", "angular_velocity", "gyro")
        if accel is None or gyro is None:
            return None
        accel = np.asarray(accel, dtype=np.float64).reshape(-1)[:3]
        gyro = np.asarray(gyro, dtype=np.float64).reshape(-1)[:3]
        if accel.size != 3 or gyro.size != 3:
            return None
        return accel, gyro
    except Exception:
        return None


def _init_debug_draw():
    """Acquire Isaac Sim's debug-draw interface when available."""
    try:
        from omni.isaac.debug_draw import _debug_draw
        return _debug_draw.acquire_debug_draw_interface()
    except Exception as exc:
        print(f"[viz] Debug draw unavailable; skipping 3D overlays ({exc})")
        return None




def _draw_wind_velocity_field(
    draw,
    center,
    base_wind_world,
    spatial_cfg,
    spacing_m=0.25,
    radius_xy_m=2.0,
    z_offsets_m=(-1.0, -0.75, -0.5, -0.25, 0.0,
                 0.25, 0.5, 0.75, 1.0),
    arrow_scale=0.35,
    max_arrows=5000,
    line_width=3.5,
    head_length_m=0.12,
    head_width_m=0.07,
):
    """Draw a dense 3D ambient wind velocity field."""

    if draw is None or max_arrows < 1:
        return
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")

    center = np.asarray(center, dtype=float).reshape(3)

    # Uniform grid spacing along X and Y.
    offsets = np.arange(
        -radius_xy_m,
        radius_xy_m + spacing_m * 0.5,
        spacing_m,
    )

    # Each Z offset creates a horizontal layer.
    points = np.array([
        center + np.array([dx, dy, dz])
        for dx in offsets
        for dy in offsets
        for dz in z_offsets_m
    ])

    # Limit arrow count if the requested grid is very large.
    if len(points) > max_arrows:
        indices = np.linspace(
            0, len(points) - 1, max_arrows, dtype=int
        )
        points = points[indices]

    wind = _wind_velocity_at_positions(
        positions_world=points,
        base_wind_world=base_wind_world,
        spatial_cfg=spatial_cfg,
    ).detach().cpu().numpy()

    starts, ends, colors, widths = [], [], [], []

    def add_line(a, b, color, width):
        starts.append(tuple(float(v) for v in a))
        ends.append(tuple(float(v) for v in b))
        colors.append(color)
        widths.append(float(width))

    shaft_color = (0.15, 0.65, 1.0, 1.0)
    head_color = (1.0, 0.75, 0.15, 1.0)

    for p, u in zip(points, wind):
        speed = np.linalg.norm(u)
        if not np.isfinite(speed) or speed < 1e-8:
            continue

        direction = u / speed
        end = p + arrow_scale * u

        # Arrow shaft.
        add_line(p, end, shaft_color, line_width)

        # Construct a perpendicular direction for the arrowhead.
        reference = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(direction, reference)) > 0.9:
            reference = np.array([1.0, 0.0, 0.0])

        side = np.cross(direction, reference)
        side /= np.linalg.norm(side)

        # V-shaped arrowhead at the endpoint.
        h = min(head_length_m, 0.4 * np.linalg.norm(end - p))
        base = end - h * direction
        wing = head_width_m * side

        add_line(base + wing, end, head_color, line_width)
        add_line(base - wing, end, head_color, line_width)

    if starts:
        draw.draw_lines(starts, ends, colors, widths)


def _draw_vector(draw, start, vector, scale, color, width=2.0):
    """Draw one scaled vector as a line in world coordinates."""
    if draw is None:
        return
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    p = np.asarray(start, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(p)):
        return
    end = p + float(scale) * v
    if np.linalg.norm(v) < 1e-8:
        return
    draw.draw_lines(
        [tuple(float(x) for x in p)],
        [tuple(float(x) for x in end)],
        [color],
        [float(width)],
    )


def _draw_disturbance_field(draw, center, disturbance_force, grid_radius=2.0,
                            spacing=1.0, vector_scale=0.12):
    """Draw a local grid of the applied disturbance FORCE (N), not wind speed.

    The current script applies one force vector to the drone. Repeating that
    vector on the grid visualizes a spatially uniform disturbance assumption;
    it is not a CFD/spatially varying wind solver.
    """
    if draw is None:
        return
    p0 = np.asarray(center, dtype=np.float64).reshape(3)
    f = np.asarray(disturbance_force, dtype=np.float64).reshape(3)
    starts, ends, colors, widths = [], [], [], []
    offsets = np.arange(-grid_radius, grid_radius + 0.5 * spacing, spacing)
    for dx in offsets:
        for dy in offsets:
            for dz in (-0.5, 0.5):
                p = p0 + np.array([dx, dy, dz], dtype=np.float64)
                if np.linalg.norm(f) < 1e-8:
                    continue
                q = p + vector_scale * f
                starts.append(tuple(p.tolist()))
                ends.append(tuple(q.tolist()))
                colors.append((0.15, 0.65, 1.0, 0.8))
                widths.append(1.5)
    if starts:
        draw.draw_lines(starts, ends, colors, widths)


def _draw_force_vectors(draw, state, mass, thrust_n, disturbance_force,
                        vector_scale=0.08):
    """Draw approximate thrust, gravity, and applied disturbance force."""
    if draw is None:
        return
    position = np.asarray(state[0:3], dtype=np.float64)
    quat = np.asarray(state[3:7], dtype=np.float64)
    R = quaternion_to_matrix(quat)  # body -> world
    thrust_world = R @ np.array([0.0, 0.0, float(thrust_n)])
    gravity_world = np.array([0.0, 0.0, -float(mass) * 9.81])
    _draw_vector(draw, position, thrust_world, vector_scale,
                 (0.1, 1.0, 0.2, 1.0), 3.0)
    _draw_vector(draw, position, gravity_world, vector_scale,
                 (1.0, 0.2, 0.2, 1.0), 3.0)
    _draw_vector(draw, position, disturbance_force, vector_scale,
                 (1.0, 0.65, 0.05, 1.0), 3.0)


# ── Ctrl+C handling (Kit's built-in handler segfaults mid-step) ─────
_stop = [False]
def _on_sigint(signum, frame):
    _stop[0] = True
    print("\n[signal] Ctrl+C — stopping after current step...", flush=True)


def _quat_world_to_body(quat, vec_world):
    """Rotate a world-frame vector into body frame: R^T @ v."""
    R = quaternion_to_matrix(quat)
    return R.T @ np.asarray(vec_world, dtype=np.float64)


def _compute_imu_from_state(state, prev_velocity, dt, gravity=9.81):
    """
    Synthetic IMU from ground-truth state.

    Parameters
    ----------
    state : (13,) array
        [pos(3), quat(4) as (w,x,y,z), lin_vel(3), ang_vel(3)]
    prev_velocity : (3,) array
        world-frame velocity of previous step
    dt : float
    gravity : float

    Returns
    -------
    accel_body : (3,) accelerometer reading in body frame (m/s²)
                 includes gravity reaction (reads 1g upward at rest)
    gyro_body  : (3,) gyroscope reading in body frame (rad/s)
    """
    quat = state[3:7]
    vel = state[7:10]
    ang_vel_body = state[10:13]

    # World-frame acceleration via finite difference
    a_world = (np.asarray(vel, dtype=np.float64) - np.asarray(prev_velocity, dtype=np.float64)) / dt

    # Specific force = a_world - g_world,  g_world = [0, 0, -gravity]
    g_world = np.array([0.0, 0.0, -gravity], dtype=np.float64)
    f_world = a_world - g_world

    # Project to body frame
    accel_body = _quat_world_to_body(quat, f_world)
    gyro_body = np.asarray(ang_vel_body, dtype=np.float64)

    return accel_body, gyro_body


# ── Drone prim path resolution ─────────────────────────────────────
def _get_drone_prim_path(drone):
    for attr in ("prim_path", "prim_paths"):
        if hasattr(drone, attr):
            v = getattr(drone, attr)
            if isinstance(v, str) and v:
                return v
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return str(v[0])

    for view_attr in ("_view", "articulation_view", "_articulation_view"):
        view = getattr(drone, view_attr, None)
        if view is None:
            continue
        for attr in ("_prim_paths", "prim_paths"):
            if hasattr(view, attr):
                v = getattr(view, attr)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return str(v[0])

    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        name = getattr(drone, "name", type(drone).__name__)
        candidates = [
            p.GetPath().pathString
            for p in stage.Traverse()
            if name in p.GetPath().pathString and "/World/envs/env_0" in p.GetPath().pathString
        ]
        if candidates:
            candidates.sort(key=len)
            return candidates[0]
    except Exception:
        pass

    return None


# ── Wind application ───────────────────────────────────────────────
def _set_wind_force(drone, wind_force):
    """Apply an external world-frame force to the UAV base link.

    Args:
        drone: Initialized OmniDrones multirotor.
        wind_force: 3-element force vector in Newtons [Fx, Fy, Fz].
    """
    device = drone.device

    forces = torch.as_tensor(
        wind_force,
        dtype=torch.float32,
        device=device,
    ).reshape(1, 3)

    torques = torch.zeros(
        (1, 3),
        dtype=torch.float32,
        device=device,
    )

    # Apply force to the existing rigid-body view.
    # Do not create or modify USD/PhysX prims here.
    drone.base_link.apply_forces_and_torques_at_pos(
        forces=forces,
        torques=torques,
        is_global=True,
    )


# ── Wind model ─────────────────────────────────────────────────────
def _unit_direction(direction):
    d = np.asarray(direction, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(d))
    if norm < 1e-8:
        raise ValueError("Wind direction must be a non-zero 3-vector.")
    return d / norm


def _wind_velocity_at_positions(
    positions_world,
    base_wind_world,
    spatial_cfg,
):
    """
    Evaluate the prescribed ambient wind field.

    Args:
        positions_world: (..., 3), world positions in metres.
        base_wind_world: (3,), background + temporal wind in m/s.
        spatial_cfg: spatial wind configuration.

    Returns:
        Wind velocity at each position, shape (..., 3), in m/s.

    gradient_matrix_per_m has units 1/s:
        wind_delta = G @ (position - reference_position)
    """
    positions = torch.as_tensor(positions_world)
    base = torch.as_tensor(
        base_wind_world,
        device=positions.device,
        dtype=positions.dtype,
    ).reshape(3)

    if not bool(spatial_cfg.enabled):
        return torch.zeros_like(positions) + base

    reference = torch.as_tensor(
        list(spatial_cfg.reference_position_xyz),
        device=positions.device,
        dtype=positions.dtype,
    ).reshape(3)

    offset = positions - reference
    mode = str(spatial_cfg.gradient_mode).lower()

    if mode == "diagonal":
        gradient = torch.as_tensor(
            list(spatial_cfg.gradient_xyz_per_m),
            device=positions.device,
            dtype=positions.dtype,
        ).reshape(3)

        wind_delta = offset * gradient

    elif mode == "matrix":
        G = torch.as_tensor(
            spatial_cfg.gradient_matrix_per_m,
            device=positions.device,
            dtype=positions.dtype,
        ).reshape(3, 3)

        # Row-vector form equivalent to G @ offset_column.
        wind_delta = offset @ G.T

    else:
        raise ValueError(
            f"Unknown wind gradient mode: {mode!r}. "
            "Use 'diagonal' or 'matrix'."
        )

    return base + wind_delta



def _wind_velocity_world(cfg, t, dt, rng, state, device, dtype):
    """
    Generate total world-frame wind velocity in m/s.

    Total wind =
        steady background + selected profile + optional gust overlay.

    Args:
        cfg: Full configuration containing cfg.wind.
        t: Simulation time in seconds.
        dt: Simulation timestep in seconds.
        rng: NumPy random generator, e.g. np.random.default_rng(seed).
        state: Persistent dictionary carried between simulation steps.
        device, dtype: PyTorch device and dtype.

    Returns:
        wind_velocity: Tensor with shape (3,), in world-frame m/s.
        state: Updated persistent state dictionary.

    Notes:
        - Dryden mode here is a first-order correlated-noise
          approximation, NOT a validated Dryden turbulence filter.
        - random mode generates independent samples each timestep.
        - gust overlay uses a smooth sin^2 pulse.
    """
    w = cfg.wind
    kind = str(w.type).lower()

    zero = np.zeros(3, dtype=np.float32)

    # 1. Persistent background wind.
    steady = np.asarray(
        getattr(w, "steady_wind_xyz", [0.0, 0.0, 0.0]),
        dtype=np.float32,
    ).reshape(3)

    # 2. Select one additional wind profile.
    profile = zero.copy()

    if kind == "none":
        pass

    elif kind == "constant":
        direction = _unit_direction(w.constant.direction)
        profile = float(w.constant.magnitude) * direction

    elif kind == "step":
        if t >= float(w.step.step_time):
            direction = _unit_direction(w.step.direction)
            profile = float(w.step.magnitude) * direction

    elif kind == "sinusoidal":
        direction = _unit_direction(w.sinusoidal.direction)
        amplitude = float(w.sinusoidal.amplitude)
        frequency = float(w.sinusoidal.frequency_hz)
        offset = float(w.sinusoidal.offset)
        phase = float(w.sinusoidal.phase_rad)

        scalar = offset + amplitude * math.sin(
            2.0 * math.pi * frequency * t + phase
        )
        profile = scalar * direction

    elif kind == "gust":
        start = float(w.gust.start_time)
        duration = max(float(w.gust.duration), 1e-8)

        if start <= t <= start + duration:
            phase = (t - start) / duration
            envelope = math.sin(math.pi * phase) ** 2
            direction = _unit_direction(w.gust.direction)

            profile = (
                float(w.gust.magnitude) * envelope * direction
            )

    elif kind == "dryden":
        # First-order correlated stochastic velocity.
        # tau_seconds is a time constant, not a spatial length scale.
        sigma = float(w.dryden.sigma)
        tau = max(float(w.dryden.tau_seconds), float(dt), 1e-6)
        alpha = math.exp(-float(dt) / tau)

        previous = np.asarray(
            state.get("dryden_velocity", zero),
            dtype=np.float32,
        ).reshape(3)

        noise = rng.normal(0.0, sigma, size=3).astype(np.float32)

        profile = (
            alpha * previous
            + math.sqrt(max(0.0, 1.0 - alpha * alpha)) * noise
        ).astype(np.float32)

        state["dryden_velocity"] = profile.copy()

    elif kind == "random":
        sigma = float(w.random.sigma)
        profile = rng.normal(
            0.0, sigma, size=3
        ).astype(np.float32)

    else:
        raise ValueError(
            f"Unknown wind.type={kind!r}. Choose none, constant, "
            "step, sinusoidal, gust, dryden, or random."
        )

    # 3. Optional independent smooth gust overlay.
    if bool(getattr(w, "gust_enabled", False)):
        start = float(w.gust_start_s)
        duration = max(float(w.gust_duration_s), 1e-8)

        if start <= t <= start + duration:
            phase = (t - start) / duration
            envelope = math.sin(math.pi * phase) ** 2

            direction = _unit_direction(
                getattr(w, "gust_direction", [1.0, 0.0, 0.0])
            )

            profile = profile + (
                float(w.gust_amplitude_mps)
                * envelope
                * direction
            )

    # 4. Sum components and convert to a device-compatible tensor.
    total_velocity = (steady + profile).astype(np.float32)

    wind_velocity = torch.as_tensor(
        total_velocity,
        device=device,
        dtype=dtype,
    )

    return wind_velocity, state


def _state_batched(drone):
    """Return drone state as shape (1, 13) — batch dim of 1, always."""
    raw = drone.get_state()
    return raw[..., :13].reshape(1, 13)


def _state_flat(drone):
    """Return drone state as a flat (13,) tensor."""
    return _state_batched(drone).reshape(-1)

# ── Altitude hold ──────────────────────────────────────────────────
def _thrust_accel_command(drone_state, target_z, kp_z, kd_z):
    """P-D on altitude; returns commanded vertical acceleration (m/s²)."""
    state = (
        drone_state.reshape(-1) if torch.is_tensor(drone_state)
        else np.asarray(drone_state, dtype=np.float64).reshape(-1)
    )
    z = float(state[2])
    vz = float(state[9])
    accel = 9.81 + kp_z * (target_z - z) - kd_z * vz
    return max(0.0, accel)


def position_hold_commands(state, target_position, cfg):
    s = np.asarray(state, dtype=np.float64).reshape(-1)

    pos = s[0:3]
    quat = s[3:7]  # w, x, y, z
    vel = s[7:10]

    pos_cfg = cfg.position
    att_cfg = cfg.attitude
    alt_cfg = cfg.altitude

    # ----- Horizontal position controller -----
    position_error = np.asarray(target_position[:2]) - pos[:2]

    kp_position_xy = float(pos_cfg.kp_position_xy)
    kp_velocity_xy = float(pos_cfg.kp_velocity_xy)
    max_velocity_xy = float(pos_cfg.max_velocity_xy)
    max_acceleration_xy = float(pos_cfg.max_acceleration_xy)

    # Position error -> desired horizontal velocity
    v_xy_des = kp_position_xy * position_error
    v_norm = np.linalg.norm(v_xy_des)

    if v_norm > max_velocity_xy:
        v_xy_des *= max_velocity_xy / max(v_norm, 1e-9)

    # Velocity error -> desired horizontal acceleration
    a_xy = kp_velocity_xy * (v_xy_des - vel[:2])

    a_norm = np.linalg.norm(a_xy)
    if a_norm > max_acceleration_xy:
        a_xy *= max_acceleration_xy / max(a_norm, 1e-9)

    # ----- Acceleration -> desired roll/pitch -----
    ax, ay = float(a_xy[0]), float(a_xy[1])

    roll_des = np.arctan2(-ay, 9.81)
    pitch_des = np.arctan2(ax, 9.81)

    w, x, y, z = quat

    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    pitch = (
        np.sign(sinp) * np.pi / 2
        if abs(sinp) >= 1.0
        else np.arcsin(sinp)
    )

    attitude_gain = float(att_cfg.gain)
    max_rate = float(att_cfg.max_rate)

    rate_x = np.clip(
        attitude_gain * (roll_des - roll), -max_rate, max_rate
    )
    rate_y = np.clip(
        attitude_gain * (pitch_des - pitch), -max_rate, max_rate
    )

    target_rate = np.array([rate_x, rate_y, 0.0], dtype=np.float32)

    # ----- Altitude controller -----
    kp_position_z = float(alt_cfg.kp_position_z)
    kp_velocity_z = float(alt_cfg.kp_velocity_z)
    max_velocity_z = float(alt_cfg.max_velocity_z)
    max_acceleration_z = float(alt_cfg.max_acceleration_z)

    z_error = float(target_position[2] - pos[2])
    vz_des = np.clip(kp_position_z * z_error, -max_velocity_z, max_velocity_z)

    # desired NET vertical acceleration
    az_net = kp_velocity_z * (vz_des - vel[2])
    az_net = np.clip(
            az_net,
            -max_acceleration_z,
            max_acceleration_z,
        )
    
    # PD velocity/altitude command
    # Convert net acceleration into thrust acceleration.
    tilt_comp = max(0.5, np.cos(roll) * np.cos(pitch))
    thrust_accel = max(0.0, (9.81 + az_net) / tilt_comp)

    return target_rate, thrust_accel


# ── Trajectory dict ────────────────────────────────────────────────
def _init_trajectory():
    return {
        "t": [],
        "position": [],
        "orientation": [],
        "velocity": [],
        "angular_velocity": [],
        "imu_source_real": [],
        "accel_imu": [],        # ← now populated
        "gyro_imu": [],         # ← now populated
        "action": [],
        "wind_force_N": [],
        "thrust_N": [],
    }


def _control_mode(cfg):
    """Return the requested experiment mode (default: feedback-controlled)."""
    mode = str(getattr(cfg, "control_mode", "controlled")).lower()
    if mode not in {"controlled", "fixed_input"}:
        raise ValueError("control_mode must be 'controlled' or 'fixed_input'")
    return mode


def _mode_output_dir(cfg):
    # Keep the two datasets separate so a run cannot overwrite the other mode.
    return Path(cfg.collection.output_dir) / _control_mode(cfg)


def _save_trajectory(traj, cfg, traj_idx):
    out_dir = _mode_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {k: np.asarray(v) for k, v in traj.items() if len(v)}
    npz_path = out_dir / f"trajectory_{traj_idx:04d}.npz"
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "experiment": str(cfg.experiment.name),
        "control_mode": _control_mode(cfg),
        "control_mode_description": (
            "3D position-hold outer loop plus rate controller, target=(0,0,1.5)m"
            if _control_mode(cfg) == "controlled"
            else "fixed motor/action command captured from the initial state; no feedback after capture"
        ),
        "seed": int(cfg.experiment.seed),
        "wind": OmegaConf.to_container(cfg.wind, resolve=True),
        "hover": OmegaConf.to_container(cfg.hover, resolve=True),
        "simulation": OmegaConf.to_container(cfg.sim, resolve=True),
        "drone_model": OmegaConf.to_container(cfg.drone_model, resolve=True),
        "num_steps": len(traj["t"]),
        "pair_id": int(getattr(cfg.collection, "pair_id", -1)),
        "timestamp": time.time(),
        "wind_force_units": "newtons",
        "imu_note": "physics IMU when available; otherwise synthetic from ground-truth state; includes gravity reaction",
    }
    json_path = out_dir / f"trajectory_{traj_idx:04d}.json"
    json_path.write_text(json.dumps(meta, indent=2))
    print(f"[save] {npz_path} ({len(traj['t'])} steps) + metadata")


# ── Plotting ──────────────────────────────────────────────────────
def _plot_trajectory(traj, cfg, traj_idx):
    """Save a 4×2 diagnostic figure per trajectory."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping. "
              "Install with: uv pip install matplotlib")
        return

    t = np.asarray(traj["t"])
    accel = np.asarray(traj["accel_imu"])
    gyro = np.asarray(traj["gyro_imu"])
    wind = np.asarray(traj["wind_force_N"])
    pos = np.asarray(traj["position"])
    vel = np.asarray(traj["velocity"])
    thrust = np.asarray(traj["thrust_N"])

    if len(t) < 2:
        print("[plot] trajectory too short — skipping")
        return

    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
    fig.suptitle(
        f"Trajectory {traj_idx:04d}  |  wind={cfg.wind.type}  |  {len(t)} steps",
        fontsize=13, fontweight="bold",
    )

    axis_labels = ["x", "y", "z"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Row 0, col 0 — accelerometer (body frame)
    for i, lbl in enumerate(axis_labels):
        axes[0, 0].plot(t, accel[:, i], color=colors[i], label=f"a_{lbl}", lw=1.2)
    axes[0, 0].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[0, 0].set_ylabel("Accel [m/s²]")
    axes[0, 0].set_title("IMU accelerometer (body frame)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    # Row 0, col 1 — velocity (world frame)
    for i, lbl in enumerate(axis_labels):
        axes[0, 1].plot(t, vel[:, i], color=colors[i], label=f"v_{lbl}", lw=1.2)
    axes[0, 1].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[0, 1].set_ylabel("Velocity [m/s]")
    axes[0, 1].set_title("Velocity (world frame)")
    axes[0, 1].legend(loc="upper right", fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # Row 1, col 0 — gyroscope (body frame)
    for i, lbl in enumerate(axis_labels):
        axes[1, 0].plot(t, gyro[:, i], color=colors[i], label=f"ω_{lbl}", lw=1.2)
    axes[1, 0].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[1, 0].set_ylabel("Gyro [rad/s]")
    axes[1, 0].set_title("IMU gyroscope (body frame)")
    axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    # Row 1, col 1 — thrust command
    axes[1, 1].plot(t, thrust, color="#8c564b", label="thrust cmd", lw=1.2)
    axes[1, 1].set_ylabel("Thrust [N]")
    axes[1, 1].set_title("Commanded collective thrust")
    axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    # Row 2, col 0 — wind force
    for i, lbl in enumerate(axis_labels):
        axes[2, 0].plot(t, wind[:, i], color=colors[i], label=f"F_{lbl}", lw=1.2)
    axes[2, 0].plot(t, np.linalg.norm(wind, axis=1),
                    color="k", linestyle="--", lw=1.0, label="|F|")
    axes[2, 0].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[2, 0].set_ylabel("Force [N]")
    axes[2, 0].set_title("Wind force (world frame)")
    axes[2, 0].legend(loc="upper right", fontsize=8)
    axes[2, 0].grid(alpha=0.3)

    # Row 2, col 1 — accelerometer magnitude
    accel_mag = np.linalg.norm(accel, axis=1)
    axes[2, 1].plot(t, accel_mag, color="#9467bd", lw=1.2, label="|a|")
    axes[2, 1].axhline(9.81, color="k", linestyle="--", lw=0.8,
                       alpha=0.6, label="1 g (9.81)")
    axes[2, 1].set_ylabel("|Accel| [m/s²]")
    axes[2, 1].set_title("Accelerometer magnitude (should be ~1 g at hover)")
    axes[2, 1].legend(loc="upper right", fontsize=8)
    axes[2, 1].grid(alpha=0.3)

    # Row 3, col 0 — position (world frame)
    for i, lbl in enumerate(axis_labels):
        axes[3, 0].plot(t, pos[:, i], color=colors[i], label=f"p_{lbl}", lw=1.2)
    axes[3, 0].set_ylabel("Position [m]")
    axes[3, 0].set_xlabel("Time [s]")
    axes[3, 0].set_title("Position (world frame)")
    axes[3, 0].legend(loc="upper right", fontsize=8)
    axes[3, 0].grid(alpha=0.3)

    # Row 3, col 1 — XY drift + altitude error
    drift_xy = np.linalg.norm(pos[:, :2] - pos[0, :2], axis=1)
    z_err = pos[:, 2] - float(cfg.hover.target_z)
    axes[3, 1].plot(t, drift_xy, color="#d62728", lw=1.2, label="XY drift")
    axes[3, 1].plot(t, np.abs(z_err), color="#7f7f7f", lw=1.2,
                    linestyle="--", label="|z error|")
    axes[3, 1].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[3, 1].set_ylabel("Distance [m]")
    axes[3, 1].set_xlabel("Time [s]")
    axes[3, 1].set_title("Hover quality: XY drift and altitude error")
    axes[3, 1].legend(loc="upper right", fontsize=8)
    axes[3, 1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_dir = _mode_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"trajectory_{traj_idx:04d}_plot_{int(time.time())}.png"
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved to {plot_path}")



def _reset_drone(drone, sim, position: list):
    """Reset the drone state without rebuilding the physics scene."""
    import torch

    # Pause physics while changing the drone state.
    sim.pause()

    pos = torch.tensor(
        [position],
        device=drone.device,
        dtype=torch.float32,
    )
    quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]],
        device=drone.device,
        dtype=torch.float32,
    )
    vel6 = torch.zeros(
        (1, 6),
        device=drone.device,
        dtype=torch.float32,
    )

    try:
        # Reuse the existing initialized articulation view.
        drone.set_world_poses(pos, quat)
        drone.set_velocities(vel6)

        print(
            f"[reset] pose and velocity set; "
            f"target position={position}",
            flush=True,
        )
    finally:
        sim.play()

def plot_wind_velocity(wind_time_log, wind_velocity_log, save_path=None):
    if not wind_time_log or not wind_velocity_log:
        print("No wind velocity data to plot.")
        return

    times = np.asarray(wind_time_log, dtype=np.float64)
    velocities = np.asarray(wind_velocity_log, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(times, velocities[:, 0], label="Wind X")
    ax.plot(times, velocities[:, 1], label="Wind Y")
    ax.plot(times, velocities[:, 2], label="Wind Z")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wind velocity (m/s)")
    ax.set_title("Wind Velocity vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────
@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="collect_data_wind_hover",
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones import init_simulation_app
    simulation_app = init_simulation_app(cfg)

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    try:
        import omni_drones.utils.scene as scene_utils
        from omni.isaac.core.simulation_context import SimulationContext
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni_drones.robots.drone import MultirotorBase
        from omni_drones.sensors.camera import Camera, PinholeCameraCfg

        device = cfg.sim.device

        # ── Simulation ────────────────────────────────────────
        sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=cfg.sim.dt,
            rendering_dt=cfg.sim.dt,
            sim_params=cfg.sim,
            backend="torch",
            device=device,
        )

        drone, controller = MultirotorBase.make(
            cfg.drone_model.name, cfg.drone_model.controller, device,
        )
        hover_position = list(cfg.hover.position)

        translations = torch.tensor(
            [hover_position],
            dtype=torch.float32,
            device=device,
        )

        drone.spawn(translations=translations)
        scene_utils.design_scene()

        # ── Camera ────────────────────────────────────────────
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
            camera_sensor.initialize(f"/World/envs/env_0/{drone.name}_*/base_link/Camera")
            print("[camera] Initialized")

        drone.initialize()
        print("\n[diag] Drone reset API availability:")
        print(f"  drone.set_world_poses   : {hasattr(drone, 'set_world_poses')}")
        print(f"  drone.set_velocities    : {hasattr(drone, 'set_velocities')}")
        print(f"  drone._view             : {hasattr(drone, '_view')}")
        if hasattr(drone, "_view"):
            v = drone._view
            print(f"  view.set_world_poses    : {hasattr(v, 'set_world_poses')}")
            print(f"  view.set_velocities     : {hasattr(v, 'set_velocities')}")
            print(f"  view.set_joint_velocities: {hasattr(v, 'set_joint_velocities')}")
            print(f"  view.set_joint_positions : {hasattr(v, 'set_joint_positions')}")
        print()

        sim.play()
        sim.render()

        # ── Mass / prim path ──────────────────────────────────
        mass = float(
            torch.as_tensor(drone.MASS_0, device=device, dtype=torch.float32)
            .reshape(-1)[0].item()
        )
        hover_force = mass * 9.81
        drone_root = _get_drone_prim_path(drone)

        if drone_root is None:
            raise RuntimeError("Could not determine drone prim path")
        
        base_link_path = f"{drone_root}/base_link"
        base_link_prim = get_prim_at_path(base_link_path)

        if base_link_prim is None or not base_link_prim.IsValid():
            base_link_prim = get_prim_at_path(drone_root)

        if base_link_prim is None or not base_link_prim.IsValid():
            raise RuntimeError(f"Could not find base_link at {base_link_path}")
        
        print(f"[drone] prim={drone_root}  mass={mass:.4f} kg  "
              f"hover_force={hover_force:.4f} N")

        # Prefer a physics-based IMU; retain synthetic fallback for compatibility.
        imu_sensor = _create_imu_sensor(base_link_path, float(cfg.sim.dt))
        debug_draw = _init_debug_draw()
        viz_enabled = bool(getattr(getattr(cfg, "visualization", {}), "enabled", True))
        force_scale = float(getattr(getattr(cfg, "visualization", {}),
                                    "force_vector_scale", 0.08))
        field_scale = float(getattr(getattr(cfg, "visualization", {}),
                                    "field_vector_scale", 0.12))

        _mode_output_dir(cfg).mkdir(parents=True, exist_ok=True)
        control_mode = _control_mode(cfg)
        print(f"[control] mode={control_mode!r}")
        if control_mode == "fixed_input":
            print("[control] action will be computed once from the initial state and then held fixed")
        else:
            print("[control] 3D position hold target=(0.0, 0.0, 1.5) m")
        print(f"[wind] type={cfg.wind.type!r}  (world-frame force, N)")
        print(f"[hover] target_z={cfg.hover.target_z} m  "
              f"kp={cfg.hover.kp_z} kd={cfg.hover.kd_z}")

        rng = np.random.default_rng(int(cfg.experiment.seed))
        target_rate = torch.zeros(1, 3, device=device)
        dt = float(cfg.sim.dt)


        controller_cfg = cfg.controller
        aero_cfg = cfg.aerodynamics
        wind_viz = cfg.visualization.wind_field
        spatial_cfg = cfg.wind.spatial
        update_every = max(1, int(wind_viz.update_every_n_steps))
        
                
        # ── Trajectory loop ───────────────────────────────────
        for traj_idx in range(int(cfg.collection.num_trajectories)):
            print(f"\n[traj {traj_idx + 1}/{cfg.collection.num_trajectories}]")

            _reset_drone(drone, sim, position=list(cfg.hover.position))

            drone_state_batched = _state_batched(drone).detach()
            init_state_np = drone_state_batched.reshape(-1).cpu().numpy()
            prev_velocity = init_state_np[7:10].copy()
            dtype = drone_state_batched.dtype
            wind_time_log = []
            wind_velocity_log = []

            traj = _init_trajectory()
            wind_state = {}
            fixed_action = None

            for step in range(int(cfg.collection.steps_per_trajectory)):
                t = step * dt

                # Wind velocity in world frame, plus optional spatial gradient at the drone's position.
                wind_world, wind_state = _wind_velocity_world(
                    cfg=cfg,
                    t=t,
                    dt=dt,
                    rng=rng,
                    state=wind_state,
                    device=device,
                    dtype=dtype,
                )

                state = drone_state_batched[0]
                velocity_world = state[7:10]  # (3,), m/s
                quat_wxyz = state[3:7]
                R_world_body = quaternion_to_matrix(quat_wxyz)  # (3, 3)

                # Optional spatial gradient overlay on the base wind velocity.
                wind_at_com = _wind_velocity_at_positions(
                        positions_world=state[0:3],
                        base_wind_world=wind_world,
                        spatial_cfg=spatial_cfg,
                    )
                wind_np = wind_world.detach().cpu().numpy().reshape(3)

                wind_time_log.append(float(t))
                wind_velocity_log.append(wind_np.copy())


                drag_world = aerodynamic_drag_world(
                    velocity_world=velocity_world,
                    wind_world=wind_at_com,
                    R_world_body=R_world_body,
                    cfg=aero_cfg,
                )

                _set_wind_force(drone, drag_world)


                # Controlled mode closes the outer loop on x/y/z position.
                # Fixed-input mode uses the initial hover action and then disables feedback.
                if control_mode == "controlled":
                    state_np = drone_state_batched.reshape(-1).cpu().numpy()
                    target_position = np.array(
                        list(cfg.hover.position),
                        dtype=np.float64,
                    )

                    target_rate_np, accel_cmd = position_hold_commands(
                        state=state_np,
                        target_position=target_position,
                        cfg=controller_cfg,
                    )

                    target_rate = torch.as_tensor(
                        target_rate_np,
                        dtype=drone_state_batched.dtype,
                        device=drone_state_batched.device,
                    ).reshape(1, 3)
                else:
                    accel_cmd = _thrust_accel_command(
                        drone_state_batched,
                        float(cfg.hover.target_z),
                        float(cfg.hover.kp_z),
                        float(cfg.hover.kd_z),
                    )
                    target_rate = torch.zeros(
                        (1, 3),
                        dtype=drone_state_batched.dtype,
                        device=drone_state_batched.device,
                    )
                target_thrust = torch.ones(1, 1, device=device) * mass * accel_cmd

                # Controlled mode recomputes action every step. In fixed_input mode,
                # capture the initial hover action once and replay it without feedback.
                if control_mode == "controlled":
                    action = controller(
                        drone_state_batched,
                        target_rate=target_rate,
                        target_thrust=target_thrust,
                    )
                elif fixed_action is None:
                    fixed_action = controller(
                        drone_state_batched,
                        target_rate=target_rate,
                        target_thrust=target_thrust,
                    ).detach().clone()
                    action = fixed_action
                else:
                    action = fixed_action
                    
                drone.apply_action(action)
                sim.step(render=True)

                # Read fresh state — batched for controller, flat for math/logging
                drone_state_batched = _state_batched(drone).detach()
                s = drone_state_batched.reshape(-1).cpu().numpy()

                # Prefer a real physics IMU reading; fall back to state-derived IMU.
                imu_reading = _read_imu_sensor(imu_sensor)
                if imu_reading is None:
                    accel_body, gyro_body = _compute_imu_from_state(
                        s, prev_velocity, dt
                    )
                    imu_is_real = False
                else:
                    accel_body, gyro_body = imu_reading
                    imu_is_real = True


                if (
                    viz_enabled
                    and debug_draw is not None
                    and wind_viz.enabled
                    and step % update_every == 0
                ):
                    debug_draw.clear_lines()

                    # Ambient wind velocity field, arrows represent m/s.
                    if wind_viz.show_velocity:
                        _draw_wind_velocity_field(
                            draw=debug_draw,
                            center=state[0:3].detach().cpu().numpy(),
                            base_wind_world=wind_world,
                            spatial_cfg=cfg.wind.spatial,
                            spacing_m=float(wind_viz.spacing_m),
                            radius_xy_m=float(wind_viz.radius_xy_m),
                            z_offsets_m=list(wind_viz.z_offsets_m),
                            arrow_scale=float(wind_viz.velocity_arrow_scale),
                            max_arrows=int(wind_viz.max_arrows)
                        )

                    # Draw forces separately: these vectors represent newtons.
                    if wind_viz.show_drag_force:
                        _draw_force_vectors(
                            debug_draw,
                            state,
                            mass,
                            float(target_thrust.item()),
                            drag_world.detach().cpu().numpy(),
                            vector_scale=force_scale,
                        )

                # Log
                traj["t"].append(t)
                traj["position"].append(s[0:3].tolist())
                traj["orientation"].append(s[3:7].tolist())
                traj["velocity"].append(s[7:10].tolist())
                traj["angular_velocity"].append(s[10:13].tolist())
                traj["imu_source_real"].append(bool(imu_is_real))
                traj["accel_imu"].append(accel_body.tolist())
                traj["gyro_imu"].append(gyro_body.tolist())
                traj["action"].append(action.detach().cpu().flatten().tolist())
                traj["wind_force_N"].append(drag_world.tolist())
                traj["thrust_N"].append(float(target_thrust.item()))

                prev_velocity = s[7:10].copy()

                if step % max(1, int(round(1.0 / dt))) == 0:
                    print(
                        f"[t={t:6.2f}s] "
                        f"p=({s[0]:+.2f},{s[1]:+.2f},{s[2]:+.2f}) m "
                        f"v=({s[7]:+.2f},{s[8]:+.2f},{s[9]:+.2f}) m/s "
                        f"a_body=({accel_body[0]:+.2f},{accel_body[1]:+.2f},"
                        f"{accel_body[2]:+.2f}) m/s² "
                        f"|a|={np.linalg.norm(accel_body):.2f} "
                        f"wind=|F|={np.linalg.norm(drag_world):.2f} N",
                        flush=True,
                    )

                if s[2] < 0.03:
                    print(f"[traj] ground hit at step {step} (z={s[2]:.3f})")
                    break

                if _stop[0]:
                    print("[stop] SIGINT — breaking out")
                    break

            _save_trajectory(traj, cfg, traj_idx)
            _plot_trajectory(traj, cfg, traj_idx)
            wind_plot_path = (
                _mode_output_dir(cfg)
                / f"wind_velocity_{traj_idx:04d}.png"
            )

            plot_wind_velocity(
                wind_time_log,
                wind_velocity_log,
                save_path=str(wind_plot_path),
            )

            if _stop[0]:
                break

    except KeyboardInterrupt:
        print("\n[stop] Keyboard interrupt.", flush=True)
    except Exception as e:
        print(f"\n[error] {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
        print("[done] Simulation closed.", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
