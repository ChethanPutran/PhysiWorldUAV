#!/usr/bin/env python3
"""
collect_data.py
================
Collect drone trajectories from OmniDrones without reinforcement learning.

Usage:
    python collect_data.py --config-name collect_data
    # or override individual parameters:
    python collect_data.py wind.type=step wind.step.magnitude=8.0

All data is saved as compressed .npz files with accompanying JSON metadata.
"""

import os
import json
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

# ----------------------------------------------------------------------
# IMPORTANT: Start Isaac Sim BEFORE importing any omni.isaac.* modules.
# ----------------------------------------------------------------------
from isaacsim import SimulationApp

# We defer OmniDrones imports until after the app is created.


def _start_sim_app(cfg: DictConfig) -> SimulationApp:
    """Create the SimulationApp with the settings from the config."""
    sim_app = SimulationApp(
        {
            "headless": cfg.simulation.headless,
            "device": cfg.simulation.device,
            "multi_gpu": False,
        }
    )
    return sim_app


def _build_env(sim_app: SimulationApp, cfg: DictConfig):
    """Import OmniDrones and build the Track environment."""
    # Import after SimulationApp is alive.
    import gymnasium as gym
    import omni_drones  # registers environments
    import omni_drones.envs  # noqa: F401 – triggers registration

    # Use Hydra config values to construct the env.
    env_id = f"Isaac-Drone-{cfg.task.name}-v0"
    print(f"[collect_data] Creating environment: {env_id}")

    env = gym.make(
        env_id,
        headless=cfg.simulation.headless,
        num_envs=cfg.task.num_envs,
        max_episode_length=cfg.task.max_episode_length,
        seed=cfg.experiment.seed,
    )
    return env


# ----------------------------------------------------------------------
# Wind application helpers
# ----------------------------------------------------------------------
def _apply_constant_wind(env, force_w, torque_w):
    """
    Apply a constant world-frame force/torque using the drone's
    permanent_wrench_composer (Isaac Lab) or a PhysX attribute.
    """
    drone = env.drone  # MultirotorBase instance

    # Preferred: use WrenchComposer if available.
    if hasattr(drone, "permanent_wrench_composer"):
        force = torch.tensor(force_w, dtype=torch.float32, device=env.device)
        torque = torch.tensor(torque_w, dtype=torch.float32, device=env.device)
        # Shape: (num_envs, 1, 3)
        force = force.unsqueeze(0).unsqueeze(0).expand(env.num_envs, -1, -1)
        torque = torque.unsqueeze(0).unsqueeze(0).expand(env.num_envs, -1, -1)
        drone.permanent_wrench_composer.set_forces_and_torques(
            forces=force, torques=torque
        )
        print(f"[wind] Constant force applied: {force_w}")
        return

    # Fallback: direct PhysX attribute (if WrenchComposer unavailable).
    try:
        from omni.isaac.core.utils.prims import get_prim_at_path
        from pxr import PhysxSchema, Gf
        root_prim = get_prim_at_path(
            f"/World/envs/env_0/{cfg.task.drone_model}/base_link"
        )
        rb_api = PhysxSchema.PhysxRigidBodyAPI(root_prim)
        rb_api.GetForceAttr().Set(Gf.Vec3f(*force_w))
        rb_api.GetTorqueAttr().Set(Gf.Vec3f(*torque_w))
        print(f"[wind] Direct PhysX force applied: {force_w}")
    except Exception as e:
        print(f"[wind] WARNING: could not apply wind – {e}")


def _apply_step_wind(env, magnitude, direction, step_time, current_step):
    """Apply a sudden gust after `step_time` seconds."""
    if current_step * env.dt < step_time:
        return  # before gust
    vec = np.array(direction, dtype=np.float32) * magnitude
    _apply_constant_wind(env, vec.tolist(), [0.0, 0.0, 0.0])


def _apply_dryden_wind(env, sigma, length_scale, dt):
    """
    Simplified Dryden turbulence: zero-mean Gaussian force with
    intensity proportional to sigma.
    """
    # For a proper Dryden model you would filter white noise.
    # Here we generate a random force each step as a stand-in.
    force = np.random.normal(0.0, scale=sigma, size=3)
    _apply_constant_wind(env, force.tolist(), [0.0, 0.0, 0.0])


# ----------------------------------------------------------------------
# Scripted trajectory generator
# ----------------------------------------------------------------------
def _scripted_action(env, step, total_steps):
    """
    Return a simple action that makes the drone follow a gentle
    circle in the XY-plane. You can replace this with any trajectory
    you like (figure-eight, straight line, etc.).
    """
    # The Track environment expects actions in its own action space.
    # For a LeePositionController the action is a 4-D vector:
    # [thrust, roll_rate, pitch_rate, yaw_rate] (normalised -1..1).
    # A minimal script: maintain altitude and a small circular motion.

    thrust = 0.7  # near hover
    # Small roll/pitch commands to create a gentle circle.
    radius = 2.0  # metres (conceptual)
    omega = 0.2   # rad/s
    t = step * env.dt
    roll_cmd = 0.1 * np.sin(omega * t)
    pitch_cmd = 0.1 * np.cos(omega * t)
    yaw_cmd = 0.0

    action = np.array([thrust, roll_cmd, pitch_cmd, yaw_cmd], dtype=np.float32)
    return action


# ----------------------------------------------------------------------
# Data logging
# ----------------------------------------------------------------------
def _init_trajectory_dict():
    return {
        "t": [],
        "position": [],
        "velocity": [],
        "orientation": [],   # quaternion
        "angular_velocity": [],
        "throttle": [],
        "action": [],
        "wind_force": [],
        "imu_accel": [],
        "imu_gyro": [],
        "camera": [],        # empty by default
    }


def _log_step(traj, env, step, action, wind_force):
    """Append one timestep of data to the trajectory dictionary."""
    drone = env.drone

    # ---- state ----
    
     = drone.get_state()  # returns a tensor
    # state layout (from OmniDrones docs):
    # [pos(3), quat(4), lin_vel(3), ang_vel(3), heading(3), up(3), throttle(1), ...]
    state_np = state.cpu().numpy().squeeze()  # remove batch dims
    traj["t"].append(step * env.dt)
    traj["position"].append(state_np[0:3].tolist())
    traj["orientation"].append(state_np[3:7].tolist())
    traj["velocity"].append(state_np[7:10].tolist())
    traj["angular_velocity"].append(state_np[10:13].tolist())
    # throttle is usually the last element for simple drones
    traj["throttle"].append(float(state_np[-1]))

    # ---- action ----
    traj["action"].append(action.tolist())

    # ---- wind ----
    traj["wind_force"].append(wind_force.tolist() if isinstance(wind_force, np.ndarray) else wind_force)

    # ---- IMU (if available) ----
    if hasattr(drone, "imu") and drone.imu is not None:
        imu = drone.imu
        traj["imu_accel"].append(imu.linear_acceleration.cpu().numpy().tolist())
        traj["imu_gyro"].append(imu.angular_velocity.cpu().numpy().tolist())

    # ---- camera (optional) ----
    if hasattr(drone, "camera") and drone.camera is not None:
        img = drone.camera.get_rgb()  # H×W×3 numpy array
        traj["camera"].append(img)


def _save_trajectory(traj, cfg, traj_idx):
    """Save a trajectory dictionary to .npz with JSON metadata."""
    out_dir = Path(cfg.collection.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert lists to numpy arrays
    np_dict = {k: np.array(v) for k, v in traj.items() if len(v) > 0}

    npz_path = out_dir / f"trajectory_{traj_idx:04d}.npz"
    np.savez_compressed(npz_path, **np_dict)

    # Metadata
    meta = {
        "experiment": cfg.experiment.name,
        "seed": cfg.experiment.seed,
        "wind": OmegaConf.to_container(cfg.wind, resolve=True),
        "simulation": OmegaConf.to_container(cfg.simulation, resolve=True),
        "task": OmegaConf.to_container(cfg.task, resolve=True),
        "num_steps": len(traj["t"]),
        "timestamp": time.time(),
    }
    json_path = out_dir / f"trajectory_{traj_idx:04d}.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[save] {npz_path}  ({len(traj['t'])} steps)  + metadata")


# ----------------------------------------------------------------------
# Main collection loop
# ----------------------------------------------------------------------
def collect(cfg: DictConfig):
    """Run the data-collection loop."""
    print("=" * 60)
    print("Starting OmniDrones data collection")
    print("=" * 60)

    # 1. Start Isaac Sim
    sim_app = _start_sim_app(cfg)

    # 2. Build environment (imports OmniDrones after app is alive)
    env = _build_env(sim_app, cfg)

    # 3. Prepare output directory
    Path(cfg.collection.output_dir).mkdir(parents=True, exist_ok=True)

    # 4. Determine wind function
    wind_type = cfg.wind.type
    if wind_type == "constant":
        wind_fn = lambda env, step: _apply_constant_wind(
            env, cfg.wind.constant.force_w, cfg.wind.constant.torque_w
        )
        wind_vec = np.array(cfg.wind.constant.force_w, dtype=np.float32)
    elif wind_type == "step":
        wind_fn = lambda env, step: _apply_step_wind(
            env,
            cfg.wind.step.magnitude,
            cfg.wind.step.direction,
            cfg.wind.step.step_time,
            step,
        )
        wind_vec = np.array(cfg.wind.step.direction, dtype=np.float32) * cfg.wind.step.magnitude
    elif wind_type == "dryden":
        wind_fn = lambda env, step: _apply_dryden_wind(
            env, cfg.wind.dryden.sigma, cfg.wind.dryden.length_scale, env.dt
        )
        wind_vec = np.array(cfg.wind.dryden.sigma, dtype=np.float32)
    else:
        wind_fn = lambda env, step: None
        wind_vec = np.zeros(3, dtype=np.float32)

    # 5. Outer loop: trajectories
    for traj_idx in range(cfg.collection.num_trajectories):
        print(f"\n[traj {traj_idx+1}/{cfg.collection.num_trajectories}]")

        # Reset environment with a deterministic seed per trajectory
        obs, info = env.reset(seed=cfg.experiment.seed + traj_idx)

        traj = _init_trajectory_dict()

        # Inner loop: simulation steps
        for step in range(cfg.collection.steps_per_trajectory):
            # Apply wind
            wind_fn(env, step)

            # Compute a scripted action
            action = _scripted_action(env, step, cfg.collection.steps_per_trajectory)

            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(
                torch.as_tensor(action, device=env.device).unsqueeze(0)
            )

            # Log data
            _log_step(traj, env, step, action, wind_vec)

            if terminated.any() or truncated.any():
                print(f"  episode ended at step {step}")
                break

        # Save the trajectory
        _save_trajectory(traj, cfg, traj_idx)

    # 6. Clean shutdown
    env.close()
    sim_app.close()
    print("\n[collect_data] Done. All trajectories saved to", cfg.collection.output_dir)


# ----------------------------------------------------------------------
# Hydra entry point
# ----------------------------------------------------------------------
@hydra.main(config_path="config", config_name="collect_data", version_base=None)
def main(cfg: DictConfig):
    # Print the full config so the run is reproducible.
    print(OmegaConf.to_yaml(cfg))
    collect(cfg)


if __name__ == "__main__":
    main()