
#!/usr/bin/env python3
"""
manual_flight.py
================
Fly a Hummingbird manually in Isaac Sim.

Features:
- Keyboard-based roll, pitch, yaw, and thrust control
- CSV logging of drone state and control commands
- RGB camera attached to the drone
- Keyboard reset
- Periodic flight status output

Thrust convention:
- thrust_accel_base and thrust_accel_step are in m/s^2.
- RateController target_thrust is a force in newtons.
- Collective thrust command = drone mass * thrust acceleration.
"""

import csv
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch


class KeyboardState:
    """Track keyboard keys currently held down."""

    def __init__(self):
        import carb
        import omni.appwindow

        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()

        self._pressed = set()
        self._reset_requested = False

        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_event
        )

    def _on_event(self, event, *args):
        key = str(event.input).split(".")[-1]
        event_type = str(event.type)

        if "KEY_PRESS" in event_type:
            self._pressed.add(key)
            print(f"[keyboard] pressed: {key}", flush=True)

            if key == "R":
                self._reset_requested = True

        elif "KEY_RELEASE" in event_type:
            self._pressed.discard(key)
            print(f"[keyboard] released: {key}", flush=True)

        return True

    def is_pressed(self, key):
        return key in self._pressed

    def consume_reset(self):
        requested = self._reset_requested
        self._reset_requested = False
        return requested


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="manual_flight",
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones import init_simulation_app

    simulation_app = init_simulation_app(cfg)

    try:
        import omni_drones.utils.scene as scene_utils
        from omni.isaac.core.simulation_context import SimulationContext
        from omni_drones.robots.drone import MultirotorBase
        from omni_drones.sensors.camera import Camera, PinholeCameraCfg

        device = cfg.sim.device

        # ------------------------------------------------------------
        # Simulation
        # ------------------------------------------------------------
        sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=cfg.sim.dt,
            rendering_dt=cfg.sim.dt,
            sim_params=cfg.sim,
            backend="torch",
            device=device,
        )

        # ------------------------------------------------------------
        # Drone
        # ------------------------------------------------------------
        drone, controller = MultirotorBase.make(
            cfg.drone_model.name,
            cfg.drone_model.controller,
            device,
        )

        translations = torch.zeros(1, 3, device=device)
        translations[:, 2] = 0.5

        drone.spawn(translations=translations)
        scene_utils.design_scene()

        # ------------------------------------------------------------
        # Camera: spawn before simulation reset
        # ------------------------------------------------------------
        camera_sensor = None

        if cfg.camera.enabled:
            camera_cfg = PinholeCameraCfg(
                sensor_tick=0,
                resolution=tuple(
                    int(value) for value in cfg.camera.resolution
                ),
                data_types=["rgb"],
            )

            camera_sensor = Camera(camera_cfg)

            camera_prim = (
                f"/World/envs/env_0/"
                f"{drone.name}_0/base_link/Camera"
            )

            camera_sensor.spawn([camera_prim])
            print(f"[camera] Spawned prim at {camera_prim}")

        # ------------------------------------------------------------
        # Reset and initialize
        # ------------------------------------------------------------
        sim.reset()

        if cfg.camera.enabled:
            camera_sensor.initialize(
                f"/World/envs/env_0/"
                f"{drone.name}_*/base_link/Camera"
            )
            print(
                "[camera] Initialized. "
                "Select it in the viewport camera dropdown."
            )

        drone.initialize()
        sim.play()

        # ------------------------------------------------------------
        # Read drone mass and prepare thrust conversion
        # ------------------------------------------------------------
        # The RateController expects target_thrust in force units (N),
        # not a normalized value between 0 and 1.
        mass = torch.as_tensor(
            drone.MASS_0, device=device, dtype=torch.float32
        ).reshape(-1)[0]

        mass_value = float(mass.item())
        hover_force = mass_value * 9.81

        print(f"[drone] Mass: {mass_value:.4f} kg")
        print(f"[drone] Nominal hover force: {hover_force:.4f} N")

        # ------------------------------------------------------------
        # CSV logging
        # ------------------------------------------------------------
        log_handle = None
        log_writer = None
        log_file = None

        if cfg.logging.enabled:
            output_dir = Path(cfg.logging.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = output_dir / f"flight_{timestamp}.csv"

            log_handle = open(
                log_file, "w", newline="", buffering=1
            )

            fieldnames = [
                "step",
                "pos_x", "pos_y", "pos_z",
                "quat_w", "quat_x", "quat_y", "quat_z",
                "vel_x", "vel_y", "vel_z",
                "ang_vel_x", "ang_vel_y", "ang_vel_z",
                "cmd_thrust_N",
                "cmd_thrust_accel_mps2",
                "cmd_roll_rate",
                "cmd_pitch_rate",
                "cmd_yaw_rate",
            ]

            log_writer = csv.DictWriter(
                log_handle, fieldnames=fieldnames
            )
            log_writer.writeheader()

            print(f"[log] Writing to {log_file}")

        # ------------------------------------------------------------
        # Keyboard and controller targets
        # ------------------------------------------------------------
        kb = KeyboardState()

        target_rate = torch.zeros(1, 3, device=device)

        thrust_accel = float(cfg.thrust_accel_base)
        target_thrust = (
            torch.ones(1, 1, device=device) * mass * thrust_accel
        )

        print("\n=== Manual Flight Control ===")
        print(
            f"  {cfg.keyboard.thrust_up}/"
            f"{cfg.keyboard.thrust_down}: "
            f"vertical thrust acceleration "
            f"(base={cfg.thrust_accel_base}, "
            f"step=±{cfg.thrust_accel_step} m/s²)"
        )
        print(
            f"  {cfg.keyboard.roll_left}/"
            f"{cfg.keyboard.roll_right}: roll rate"
        )
        print(
            f"  {cfg.keyboard.pitch_forward}/"
            f"{cfg.keyboard.pitch_backward}: pitch rate"
        )
        print(
            f"  {cfg.keyboard.yaw_left}/"
            f"{cfg.keyboard.yaw_right}: yaw rate"
        )
        print(f"  {cfg.keyboard.reset}: reset drone")
        print("Click the Isaac Sim viewport first, then use the keys.")
        print("Press Ctrl+C to stop.\n")

        step = 0
        last_state = None

        drone_state = (
            drone.get_state()[..., :13].squeeze(0).detach()
        )

        # ------------------------------------------------------------
        # Flight loop
        # ------------------------------------------------------------
        while simulation_app.is_running():
            # Convert the configured acceleration command to thrust force.
            thrust_accel = float(cfg.thrust_accel_base)

            if kb.is_pressed(cfg.keyboard.thrust_up):
                thrust_accel += float(cfg.thrust_accel_step)

            if kb.is_pressed(cfg.keyboard.thrust_down):
                thrust_accel -= float(cfg.thrust_accel_step)

            # Prevent negative collective thrust.
            thrust_accel = max(0.0, thrust_accel)

            # Force = mass * commanded acceleration.
            target_thrust = (
                torch.ones(1, 1, device=device)
                * mass
                * thrust_accel
            )

            # Body-rate commands.
            roll_rate = (
                (-0.8 if kb.is_pressed(cfg.keyboard.roll_left) else 0.0)
                + (0.8 if kb.is_pressed(cfg.keyboard.roll_right) else 0.0)
            )

            pitch_rate = (
                (0.8 if kb.is_pressed(cfg.keyboard.pitch_forward) else 0.0)
                + (
                    -0.8
                    if kb.is_pressed(cfg.keyboard.pitch_backward)
                    else 0.0
                )
            )

            yaw_rate = (
                (-0.8 if kb.is_pressed(cfg.keyboard.yaw_left) else 0.0)
                + (0.8 if kb.is_pressed(cfg.keyboard.yaw_right) else 0.0)
            )

            # Reset the drone when R is pressed.
            if kb.consume_reset():
                drone._reset_idx(
                    torch.tensor([0], device=device)
                )

                drone_state = (
                    drone.get_state()[..., :13]
                    .squeeze(0)
                    .detach()
                )

                step = 0
                last_state = None
                print("[reset] Drone reset requested.", flush=True)

            target_rate[0, 0] = roll_rate
            target_rate[0, 1] = pitch_rate
            target_rate[0, 2] = yaw_rate

            # Calculate the controller action.
            action = controller(
                drone_state,
                target_rate=target_rate,
                target_thrust=target_thrust,
            )

            drone.apply_action(action)
            sim.step(render=True)

            # --------------------------------------------------------
            # State logging
            # --------------------------------------------------------
            state = (
                drone.get_state()[..., :13]
                .reshape(-1, 13)[0]
                .detach()
                .cpu()
                .numpy()
            )

            last_state = state

            if log_writer is not None:
                log_writer.writerow({
                    "step": step,
                    "pos_x": state[0],
                    "pos_y": state[1],
                    "pos_z": state[2],
                    "quat_w": state[3],
                    "quat_x": state[4],
                    "quat_y": state[5],
                    "quat_z": state[6],
                    "vel_x": state[7],
                    "vel_y": state[8],
                    "vel_z": state[9],
                    "ang_vel_x": state[10],
                    "ang_vel_y": state[11],
                    "ang_vel_z": state[12],
                    "cmd_thrust_N": float(target_thrust[0, 0].item()),
                    "cmd_thrust_accel_mps2": thrust_accel,
                    "cmd_roll_rate": roll_rate,
                    "cmd_pitch_rate": pitch_rate,
                    "cmd_yaw_rate": yaw_rate,
                })

            drone_state = (
                drone.get_state()[..., :13]
                .squeeze(0)
                .detach()
            )

            step += 1

            # Print status every 50 simulation steps.
            if step % 50 == 0 and last_state is not None:
                print(
                    f"[status] step={step} "
                    f"pos=({last_state[0]:.2f},"
                    f"{last_state[1]:.2f},"
                    f"{last_state[2]:.2f}) "
                    f"vel_z={last_state[9]:.2f} m/s "
                    f"thrust={target_thrust[0, 0].item():.3f} N "
                    f"accel_cmd={thrust_accel:.2f} m/s²",
                    flush=True,
                )

            if step % 10 == 0:
                print(
                    f"[debug] z={drone.pos[..., 2].item():.3f} "
                    f"vz={drone.vel[..., 2].item():.3f} "
                    f"thrust_N={target_thrust.item():.3f} "
                    f"action={action.detach().cpu().flatten().tolist()}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n[stop] Keyboard interrupt.", flush=True)

    finally:
        if log_handle is not None:
            log_handle.close()
            print(f"[log] Data saved to {log_file}")

        simulation_app.close()
        print("[done] Simulation closed.")


if __name__ == "__main__":
    main()
