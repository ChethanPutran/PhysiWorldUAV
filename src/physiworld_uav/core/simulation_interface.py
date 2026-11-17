"""Thin, reusable wrapper around the OmniDrones / Isaac Sim simulation.

This module extracts the simulator lifecycle that was previously inlined in
``collect_data_wind_hover.py`` into a class with a stable API. It does **not**
own the control law, the wind field, or the aerodynamic model — those stay in
their own modules. Its sole responsibility is:

    spawn → reset → get state → apply action → step → apply external force → close

Design goals
------------
* **Lazy imports**: importing this module must not require Isaac Sim / OmniDrones
  at import time, so tests and analysis notebooks can import it freely.
* **Dependency injection**: the three most version-sensitive operations
  (state extraction, reset, force application) accept optional callables. Pass
  the tested helpers from ``collect_data_wind_hover.py`` if the defaults do not
  match your installed OmniDrones version.
* **Explicit units and frames**: every public method documents its frame and
  units. No hidden coordinate conversions.
* **No hidden physics**: gravity, motor thrust, and rotor dynamics are handled
  by the simulator; this class only injects *additional* external forces.

State layout (per drone, 13 floats): see ``state_extractor``.
"""
from __future__ import annotations

import logging
import signal
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from .state_extractor import (
    GRAVITY_MPS2,
    NUM_STATE,
    compute_imu_from_state,
)

logger = logging.getLogger(__name__)

# Type aliases for the injectable callbacks.
StateFn = Callable[[Any], torch.Tensor]
ResetFn = Callable[[Any, Any, list], None]


# ---------------------------------------------------------------------------
# Signal handling (module-level, mirrors the legacy script)
# ---------------------------------------------------------------------------

_STOP: list[bool] = [False]


def _on_sigint(signum, frame):  # noqa: ANN001
    _STOP[0] = True
    logger.warning("SIGINT/SIGTERM received; will stop after the current step.")


def install_signal_handlers() -> None:
    """Idempotently install SIGINT/SIGTERM handlers that set ``_STOP``."""
    try:
        signal.signal(signal.SIGINT, _on_sigint)
        signal.signal(signal.SIGTERM, _on_sigint)
    except ValueError:
        # Not in the main thread (e.g. pytest worker); skip silently.
        pass


def stop_requested() -> bool:
    return _STOP[0]


def reset_stop_flag() -> None:
    _STOP[0] = False


# ---------------------------------------------------------------------------
# Default implementations (used when no override is supplied)
# ---------------------------------------------------------------------------


def _default_state_fn(drone: Any) -> torch.Tensor:
    """Return the batched state tensor ``(num_drones, 13)``.

    OmniDrones' ``get_state`` is documented to return the 13-element layout
    (position, quaternion, linear velocity, angular velocity) for the drone
    kinematic state. Any trailing fields (heading, up vector, throttle, rotor
    speeds) are sliced away.
    """
    state = drone.get_state()
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state)
    if state.shape[-1] < NUM_STATE:
        raise RuntimeError(
            f"drone.get_state() returned last dim {state.shape[-1]}, "
            f"expected >= {NUM_STATE}. Pass a custom ``state_fn``."
        )
    return state[..., :NUM_STATE]


def _default_reset_fn(drone: Any, sim: Any, position: list[float]) -> None:
    """Reset drone pose and velocities to ``position`` with identity attitude.

    Uses the drone-level API, which is what ``MultirotorBase`` exposes in
    OmniDrones 4.1:

    * ``drone.set_world_poses(positions, orientations)``
    * ``drone.set_velocities(velocities_6d)``  — **single ``(n, 6)`` tensor**

    Zero linear and angular velocities. Runs inside ``sim.pause()`` /
    ``sim.play()`` so that pose/velocity writes do not race the physics
    step, matching ``collect_data_wind_hover._reset_drone``.
    """
    import torch  # local; this module is importable without torch in CI

    device = getattr(drone, "device", None)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pos = torch.tensor([position], dtype=torch.float32, device=device)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]],
                                 dtype=torch.float32, device=device)
    vel6 = torch.zeros((1, 6), dtype=torch.float32, device=device)

    paused = False
    if sim is not None and hasattr(sim, "pause"):
        sim.pause()
        paused = True
    try:
        if hasattr(drone, "set_world_poses") and hasattr(drone, "set_velocities"):
            drone.set_world_poses(pos, identity_quat)
            drone.set_velocities(vel6)
        else:
            # Fall back to a view-style API when the drone object is wrapped.
            view = getattr(drone, "_view", None)
            if view is None:
                raise RuntimeError(
                    "Drone exposes neither set_world_poses nor _view; "
                    "cannot reset."
                )
            view.set_world_poses(positions=pos, orientations=identity_quat)
            view.set_velocities(velocities=vel6[:, :3].contiguous(),
                                angular_velocities=vel6[:, 3:].contiguous())
    finally:
        if paused:
            sim.play()


def _enable_isaac_extensions(
    extensions: tuple[str, ...] = (
        "omni.isaac.core",
        "omni.isaac.sensor",        # for IMU
        "omni.isaac.debug_draw",    # for debug overlays
    ),
) -> None:
    """Enable Isaac Sim extensions required by OmniDrones.

    ``init_simulation_app`` launches SimulationApp with a minimal config and
    does not guarantee that ``omni.isaac.core`` is active. We enable the
    required extensions via the low-level extension manager before importing
    any ``omni.isaac.*`` module.

    Uses ``set_extension_enabled_immediate`` (not the higher-level
    ``enable_extension`` helper) so that this works even when
    ``omni.isaac.core.utils.extensions`` is not yet importable.
    """
    try:
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
    except Exception as exc:
        logger.warning("Cannot access extension manager (%s); "
                       "extensions must be enabled manually", exc)
        return

    for ext in extensions:
        try:
            if manager.is_extension_enabled(ext):
                logger.debug("Extension already enabled: %s", ext)
                continue
            manager.set_extension_enabled_immediate(ext, True)
            logger.info("Enabled Isaac Sim extension: %s", ext)
        except Exception as exc:
            # Sensor / debug_draw are optional; core is mandatory.
            if ext == "omni.isaac.core":
                raise RuntimeError(
                    f"Failed to enable required extension {ext!r}. "
                    f"Verify the Isaac Sim installation includes "
                    f"'{ext}'. Original error: {exc}"
                ) from exc
            logger.warning("Could not enable optional extension %s: %s", ext, exc)

    # One update lets the newly-enabled extensions initialise their Python
    # modules before the next import.
    try:
        omni.kit.app.get_app().update()
    except Exception:
        pass

def _get_drone_prim_path(drone: Any) -> Optional[str]:
    """Best-effort discovery of the drone root prim path."""
    for attr in ("prim_paths", "_prim_paths"):
        paths = getattr(drone, attr, None)
        if paths:
            return str(paths[0])
    view = getattr(drone, "_view", None)
    for attr in ("prim_paths", "_prim_paths"):
        paths = getattr(view, attr, None)
        if paths:
            return str(paths[0])
    return None


# ---------------------------------------------------------------------------
# Configuration container
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """Minimal, simulator-facing subset of a Hydra/OmegaConf config."""

    dt: float = 1.0 / 200.0
    device: str = "cuda:0"
    render: bool = True
    headless: bool | None = None              # NEW: None → derive from `render`
    drone_model: str = "Hummingbird"
    controller: str = "RateController"
    drone_prim_name: str | None = None
    init_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.5])
    sim_params: dict | None = None
    camera_enabled: bool = False
    camera_resolution: tuple[int, int] = (640, 480)

    # Physics pipeline (Run 4, IMU decision (a)):
    #   "gpu_dynamics_cpu_readback" — PhysX GPU dynamics + GPU broadphase, results
    #       read back to the CPU each step (Isaac IMUSensor needs the readback);
    #       tensor views live on the CPU.
    #   "gpu" — Isaac default GPU pipeline (no readback): IMUSensor returns
    #       gravity-only frames (verified), so it is rejected for data generation.
    #   "cpu" — CPU dynamics.
    physics_pipeline: str = "gpu_dynamics_cpu_readback"
    enhanced_determinism: bool = True
    linear_damping: float | None = 0.0         # D6: OmniDrones default 0.2 (hidden drag)
    angular_damping: float | None = 0.0

    @property
    def tensor_device(self) -> str:
        return self.device if self.physics_pipeline == "gpu" else "cpu"

    # IMU
    imu_enabled: bool = True
    imu_frequency_hz: int | None = None
    imu_read_gravity: bool = True
    imu_filter_size: int = 1

    # Original Hydra config (kept for OmniDrones / SimulationContext).
    hydra_config: Any | None = None

    @property
    def effective_headless(self) -> bool:
        """Headless flag: explicit value, else `not render`, else True."""
        if self.headless is not None:
            return bool(self.headless)
        return not bool(self.render)

    @classmethod
    def from_hydra(cls, cfg: Any) -> "SimulationConfig":
        """Build from an OmegaConf DictConfig (all fields optional)."""
        def _get(section: Any, key: str, default: Any) -> Any:
            if section is None:
                return default
            try:
                if hasattr(section, "get"):
                    v = section.get(key, default)
                    return default if v is None else v
                return getattr(section, key, default)
            except Exception:
                return default

        sim = getattr(cfg, "sim", None) if hasattr(cfg, "get") else None
        drone = getattr(cfg, "drone_model", None) if hasattr(cfg, "get") else None
        hover = getattr(cfg, "hover", None) if hasattr(cfg, "get") else None
        camera = getattr(cfg, "camera", None) if hasattr(cfg, "get") else None
        imu = getattr(cfg, "imu", None) if hasattr(cfg, "get") else None

        res = _get(camera, "resolution", (640, 480))
        camera_enabled = bool(_get(camera, "enabled", False))

        # Headless: prefer explicit keys; else `not render`; else True.
        headless = _get(sim, "headless", None)
        if headless is None:
            headless = _get(getattr(cfg, "app", None) if hasattr(cfg, "get") else None,
                            "headless", None)

        return cls(
            dt=float(_get(sim, "dt", 1.0 / 200.0)),
            device=str(_get(sim, "device", "cuda:0")),
            render=bool(_get(sim, "render", False)),       # default False for batch runs
            headless=(bool(headless) if headless is not None else None),
            drone_model=str(_get(drone, "name", "Hummingbird")),
            controller=str(_get(drone, "controller", "RateController")),
            init_position=list(_get(hover, "position", [0.0, 0.0, 1.5])),
            camera_enabled=camera_enabled,
            camera_resolution=tuple(int(v) for v in res),
            imu_enabled=bool(_get(imu, "enabled", True)),
            imu_frequency_hz=(int(_get(imu, "frequency_hz", 0))
                              if _get(imu, "frequency_hz", None) else None),
            imu_read_gravity=bool(_get(imu, "read_gravity", True)),
            imu_filter_size=int(_get(imu, "filter_size", 1)),
            hydra_config=cfg,
        )

    # ------------------------------------------------------------------
    def sim_app_config(self) -> dict:
        """Return the dict that OmniDrones' ``init_simulation_app`` expects.

        ``init_simulation_app`` reads ``cfg["headless"]``. We merge any
        surviving keys from the original Hydra config with guaranteed
        defaults so this always works, regardless of how the caller built
        the config.
        """
        base: dict = {}

        # Preserve any top-level keys from the original Hydra config that
        # downstream code might read (e.g. ``num_envs``). Use a plain dict so
        # ``cfg["headless"]`` works unconditionally.
        hc = self.hydra_config
        if hc is not None:
            try:
                from omegaconf import DictConfig, OmegaConf
                if isinstance(hc, DictConfig):
                    resolved = OmegaConf.to_container(hc, resolve=True) or {}
                    if isinstance(resolved, dict):
                        base.update(resolved)
                elif isinstance(hc, dict):
                    base.update(hc)
            except Exception:
                pass

        base.setdefault("headless", self.effective_headless)
        base.setdefault("anti_aliasing", 1)
        return base

    
# ---------------------------------------------------------------------------
# SimulationInterface
# ---------------------------------------------------------------------------


class SimulationInterface:
    """Lifecycle wrapper for a single-drone OmniDrones simulation.

    Example
    -------
    >>> sim = SimulationInterface.from_hydra(cfg)
    >>> try:
    ...     sim.reset(position=[0.0, 0.0, 1.5])
    ...     state = sim.get_state()                       # (1, 13) tensor
    ...     action = sim.compute_action(target_rate, target_thrust)
    ...     sim.apply_action(action)
    ...     sim.set_external_wrench(force_W, torque_W)   # before step, every step
    ...     sim.step()
    ...     flat = sim.get_state_np()                     # (13,) numpy
    ... finally:
    ...     sim.close()
    """

    def __init__(
        self,
        cfg: SimulationConfig,
        *,
        state_fn: StateFn | None = None,
        reset_fn: ResetFn | None = None,
    ) -> None:
        self.cfg = cfg
        self._state_fn = state_fn or _default_state_fn
        self._reset_fn = reset_fn or _default_reset_fn
        self._injector = None

        self._app = None
        self._sim = None
        self._drone = None
        self._controller = None
        self._camera = None
        self._imu = None
        self._mass: float = float("nan")
        self._drone_root: Optional[str] = None
        self._base_link_path: Optional[str] = None
        self._closed = False

    # -- construction ------------------------------------------------------

    @classmethod
    def from_hydra(cls, cfg: Any, **kwargs: Any) -> "SimulationInterface":
        return cls(SimulationConfig.from_hydra(cfg), **kwargs)


    def pause(self) -> None:
        """Pause physics if the simulation has been initialised."""
        if self._sim is not None and hasattr(self._sim, "pause"):
            self._sim.pause()


    def play(self) -> None:
        """Resume physics if the simulation has been initialised."""
        if self._sim is not None and hasattr(self._sim, "play"):
            self._sim.play()

    def _setup(self) -> None:
        """Import simulators lazily, build the scene, and initialise the drone.

        Critical ordering
        -----------------
        ``init_simulation_app(cfg)`` must run **before** any import that touches
        ``omni.isaac.*`` or ``omni.kit.*``. Those extensions are only importable
        after ``SimulationApp`` initialises the carb extension cache — exactly the
        pattern used by the original ``collect_data_wind_hover.py``::

            from omni_drones import init_simulation_app
            simulation_app = init_simulation_app(cfg)   # first
            try:
                import omni_drones.utils.scene as scene_utils   # then these
                from omni.isaac.core.simulation_context import SimulationContext
                ...

        ``init_simulation_app`` receives the **full Hydra DictConfig**, not this
        class's ``SimulationConfig``. Using the dataclass here fails because
        ``init_simulation_app`` reads ``cfg.headless``, ``cfg.sim.device``, and
        viewer/window fields that do not exist on the dataclass.
        """
        install_signal_handlers()

        from omni_drones import init_simulation_app  # lazy
        cfg = self.cfg

        # 1. Launch the app ------------------------------------------------
        self._app = init_simulation_app(cfg.sim_app_config())

        # 2. Enable extensions BEFORE any omni.isaac.* import ---------------
        _enable_isaac_extensions()

        # 3. Now safe to import OmniDrones scene utilities and Isaac core ----
        import omni_drones.utils.scene as scene_utils  # lazy
        from omni.isaac.core.simulation_context import SimulationContext  # lazy
        from omni.isaac.core.utils.prims import get_prim_at_path  # lazy
        from omni_drones.robots.drone import MultirotorBase  # lazy

        # 4. SimulationContext ---------------------------------------------
        if cfg.physics_pipeline not in ("gpu_dynamics_cpu_readback", "gpu", "cpu"):
            raise ValueError(f"unknown physics_pipeline {cfg.physics_pipeline!r}")
        self._sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=cfg.dt,
            rendering_dt=cfg.dt,
            sim_params=cfg.sim_params,
            backend="torch",
            device=cfg.tensor_device,
        )
        pc = self._sim.get_physics_context()
        if cfg.physics_pipeline == "gpu_dynamics_cpu_readback":
            # applied before the scene is reset/initialised so the flags take effect
            if ":" in cfg.device:
                import carb
                carb.settings.get_settings().set_int("/physics/cudaDevice", int(cfg.device.split(":")[1]))
            pc.enable_gpu_dynamics(True)
            pc.set_broadphase_type("GPU")
        if cfg.enhanced_determinism:
            pc._physx_scene_api.CreateEnableEnhancedDeterminismAttr().Set(True)

        # 5. Spawn the drone -----------------------------------------------
        self._drone, self._controller = MultirotorBase.make(
            cfg.drone_model, cfg.controller, cfg.tensor_device,
        )
        # D6: OmniDrones applies PhysX linear/angular damping 0.2 to every body at
        # spawn (a hidden, wind-unaware drag). Override before spawn.
        if cfg.linear_damping is not None:
            self._drone.rigid_props.linear_damping = float(cfg.linear_damping)
        if cfg.angular_damping is not None:
            self._drone.rigid_props.angular_damping = float(cfg.angular_damping)
        translations = torch.tensor(
            [cfg.init_position], dtype=torch.float32, device=cfg.tensor_device,
        )
        self._drone.spawn(translations=translations)
        scene_utils.design_scene()

        # 5b. IMU prim is created BEFORE sim.reset() (Run 4): created after the
        # reset, the sensor plugin never registers the rigid body and every
        # reading is invalid (physics_step stays 0) on both CPU and GPU pipelines.
        self._imu = None
        if cfg.imu_enabled:
            spawned_root = f"/World/envs/env_0/{self._drone.name}_0"
            self._imu = _create_imu_sensor(
                base_link_path=f"{spawned_root}/base_link",
                dt=cfg.dt,
                frequency_hz=cfg.imu_frequency_hz or max(1, int(round(1.0 / cfg.dt))),
                filter_size=cfg.imu_filter_size,
                initialize=False,
            )

        self._sim.reset()

        if cfg.camera_enabled:
            self._setup_camera()

        self._drone.initialize()
        from physiworld_uav.forces.injection import install as _install_injector
        self._injector = _install_injector(self._drone)

        # 6. Live mass & prim paths ----------------------------------------
        self._mass = float(
            torch.as_tensor(self._drone.MASS_0, device=cfg.tensor_device, dtype=torch.float32)
            .reshape(-1)[0].item()
        )

        self._drone_root = _get_drone_prim_path(self._drone)
        if self._drone_root is None:
            raise RuntimeError("Could not determine drone prim path.")
        self._base_link_path = f"{self._drone_root}/base_link"

        prim = get_prim_at_path(self._base_link_path)
        if prim is None or not prim.IsValid():
            prim = get_prim_at_path(self._drone_root)
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"Could not find base_link at {self._base_link_path}")

        # 7. Physics IMU (prim created before the reset; initialise now) ----
        if self._imu is not None and not _initialize_imu(self._imu):
            self._imu = None
        if self._imu is None:
            logger.warning(
                "IMU sensor unavailable — falling back to state-derived IMU. "
                "Check that the base_link prim has RigidBodyAPI and that "
                "omni.isaac.sensor is enabled."
            )
        else:
            logger.info("IMU sensor created at %s/Imu_Sensor (%.1f Hz)",
                        self._base_link_path,
                        cfg.imu_frequency_hz or max(1, int(round(1.0 / cfg.dt))))

        self._sim.play()
        if cfg.render:
            self._sim.render()

        logger.info(
            "Simulation ready: drone=%s prim=%s mass=%.4f kg headless=%s",
            cfg.drone_model, self._drone_root, self._mass, cfg.effective_headless,
        )

    def _setup_camera(self) -> None:
        from omni_drones.sensors.camera import Camera, PinholeCameraCfg  # lazy

        cam_cfg = PinholeCameraCfg(
            sensor_tick=0,
            resolution=self.cfg.camera_resolution,
            data_types=["rgb"],
        )
        self._camera = Camera(cam_cfg)
        cam_prim = f"/World/envs/env_0/{self._drone.name}_0/base_link/Camera"
        self._camera.spawn([cam_prim])
        self._camera.initialize(f"/World/envs/env_0/{self._drone.name}_*/base_link/Camera")

    def ensure_ready(self) -> None:
        """Idempotent lazy setup: call before the first ``reset``/``step``."""
        if self._sim is None:
            self._setup()

    # -- core lifecycle ----------------------------------------------------

    def reset(self, position: list[float] | None = None) -> None:
        """Teleport the drone to ``position`` (default ``cfg.init_position``)."""
        self.ensure_ready()
        pos = list(position) if position is not None else list(self.cfg.init_position)
        self._reset_fn(self._drone, self._sim, pos)

    def get_state(self) -> torch.Tensor:
        """Batched state tensor ``(num_drones, 13)``, detached."""
        self.ensure_ready()
        return self._state_fn(self._drone).detach()

    def get_state_np(self) -> np.ndarray:
        """Flat state ``(13,)`` for the first drone, as float64 numpy."""
        return self.get_state().reshape(-1).cpu().numpy().astype(np.float64)

    def compute_action(
        self,
        target_rate: torch.Tensor | None = None,
        target_thrust: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Invoke the OmniDrones controller on the current state.

        ``target_rate`` has shape ``(1, 3)`` in rad/s (body frame);
        ``target_thrust`` has shape ``(1, 1)`` in Newtons (total).
        """
        self.ensure_ready()
        state = self.get_state()
        return self._controller(
            state,
            target_rate=target_rate,
            target_thrust=target_thrust,
        )

    def apply_action(self, action: torch.Tensor) -> None:
        """Submit rotor commands to the drone (shape ``(1, num_rotors)``)."""
        self.ensure_ready()
        self._drone.apply_action(action)

    # -- episode-runner backend API (experiments/runner.py) ----------------
    def reset_episode(self, position: np.ndarray, quat_wxyz: np.ndarray,
                      velocity_W: np.ndarray, angular_velocity_W: np.ndarray,
                      throttle: np.ndarray) -> None:
        """Full-state reset including the motor state (plan §5-A13).

        ``_reset_idx`` clears OmniDrones' cached buffers; it sets the trim
        throttle from the *base-link* mass, so the throttle is then overwritten
        with the caller's value (computed from the true total mass).
        ``angular_velocity_W`` must be world-frame (PhysX convention).
        """
        self.ensure_ready()
        drone = self._drone
        dev = torch.device(self.cfg.tensor_device)
        drone._reset_idx(torch.zeros(1, dtype=torch.long, device=dev), train=False)
        drone.throttle[:] = torch.as_tensor(throttle, dtype=torch.float32, device=dev).reshape(drone.throttle.shape)
        pos = torch.as_tensor(np.asarray(position, dtype=np.float32).reshape(1, 3), device=dev)
        quat = torch.as_tensor(np.asarray(quat_wxyz, dtype=np.float32).reshape(1, 4), device=dev)
        vel6 = torch.as_tensor(np.concatenate([velocity_W, angular_velocity_W]).astype(np.float32).reshape(1, 6),
                               device=dev)
        self.pause()
        try:
            drone.set_world_poses(pos, quat)
            drone.set_velocities(vel6)
            # rotor joints are free about the rotor axis and drift between episodes;
            # not resetting them made identical episodes differ by up to 3e-2 (Run 4)
            view = drone._view
            if getattr(view, "num_dof", 0):
                zeros = torch.zeros(1, view.num_dof, dtype=torch.float32, device=dev)
                view.set_joint_positions(zeros)
                view.set_joint_velocities(zeros)
        finally:
            self.play()

    def bind_domain(self, params, dt: float) -> dict[str, Any]:
        """Write a domain's parameters into PhysX/OmniDrones and return the realised values."""
        from physiworld_uav.drones.sim_binding import bind, realised
        self.ensure_ready()
        bind(self._drone, params, dt)
        return realised(self._drone, dt)

    def throttle(self) -> torch.Tensor:
        """(1, R) rotor throttle after the motor lag (= Ω/Ω_max)."""
        return self._drone.throttle.reshape(1, -1).detach()

    def apply_rotor_command(self, cmd: torch.Tensor) -> None:
        """Rotor commands (1, R) in [-1, 1] (OmniDrones convention)."""
        self._drone.apply_action(cmd.reshape(1, 1, -1))

    def set_external_wrench(self, force_W: torch.Tensor, torque_W: torch.Tensor) -> None:
        """World-frame external force [N] / torque [N m] for the next physics step.

        Injected into OmniDrones' single base-link call (forces/injection.py).
        Must be called every step it should act; it is cleared after use.
        Gravity and rotor thrust/torque are simulator-native and must not be included.
        """
        self.ensure_ready()
        self._injector.set(force_W, torque_W)

    def last_external_wrench(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        return None if self._injector is None else self._injector.last_applied

    def step(self, render: bool | None = None) -> None:
        """Advance physics by one ``dt``."""
        self.ensure_ready()
        self._sim.step(render=self.cfg.render if render is None else render)

    # -- sensor / metadata accessors ---------------------------------------

    def read_imu(self) -> tuple[np.ndarray, np.ndarray, bool] | None:
        """Return ``(accel_body, gyro_body, is_real)`` or ``None``.

        ``accel_body`` is **specific force** (proper acceleration) in the body/IMU
        frame, including gravity when ``cfg.imu_read_gravity`` is True. This is the
        same convention the state-derived fallback uses.

        Returns ``None`` when no sensor is available or the latest frame is
        invalid; callers should then fall back to ``compute_imu_from_state``.
        """
        if self._imu is None:
            return None

        frame = _read_imu_frame(self._imu, self.cfg.imu_read_gravity)
        if frame is None:
            return None

        # Freshness gate (plan §5-A12): Isaac's IMU frame has no validity key, so a
        # frame counts as valid only if its physics step advanced since the last read.
        step = frame.get("physics_step")
        step = int(step.item() if hasattr(step, "item") else step) if step is not None else None
        if step is None or step <= getattr(self, "_imu_last_step", 0):
            return None
        self._imu_last_step = step

        accel = _extract_vec(
            frame,
            ("lin_acc", "linear_acceleration", "lin_acc_b", "linear_acceleration_b"),
        )
        gyro = _extract_vec(
            frame,
            ("ang_vel", "angular_velocity", "ang_vel_b", "angular_velocity_b"),
        )
        if accel is None or gyro is None:
            logger.debug("IMU frame missing acceleration/gyro keys: %s", list(frame))
            return None

        return accel.astype(np.float64), gyro.astype(np.float64), True


    def read_imu_frame(self) -> dict | None:
        """Raw sensor frame, for diagnostics and logging."""
        if self._imu is None:
            return None
        return _read_imu_frame(self._imu, self.cfg.imu_read_gravity)

    def imu_fallback(
        self, state_np: np.ndarray, prev_velocity_world: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper for :func:`compute_imu_from_state`."""
        return compute_imu_from_state(state_np, prev_velocity_world, self.cfg.dt)

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def dt(self) -> float:
        return self.cfg.dt

    @property
    def device(self) -> torch.device:
        """Device of the simulator tensor views (CPU when results are read back)."""
        return torch.device(self.cfg.tensor_device)

    def physics_settings(self) -> dict[str, Any]:
        """Settings read back from the running simulation (plan §1-V4)."""
        import carb
        from pxr import PhysxSchema, UsdPhysics
        from omni.isaac.core.utils.prims import get_prim_at_path
        pc = self._sim.get_physics_context()
        st = carb.settings.get_settings()
        out: dict[str, Any] = {
            "pipeline": self.cfg.physics_pipeline,
            "gpu_dynamics": bool(pc.is_gpu_dynamics_enabled()),
            "broadphase": str(pc.get_broadphase_type()),
            "suppress_readback": bool(st.get_as_bool("/physics/suppressReadback")),
            "cuda_device": int(st.get_as_int("/physics/cudaDevice")),
            "tensor_device": self.cfg.tensor_device,
            "physics_dt": float(pc.get_physics_dt()),
            "solver_type": str(pc.get_solver_type()),
            "gravity_W": _gravity_vector(pc.get_gravity()),
            "enhanced_determinism": bool(
                pc._physx_scene_api.GetEnableEnhancedDeterminismAttr().Get() or False),
        }
        damping = {}
        for link in ("base_link", "rotor_0", "rotor_1", "rotor_2", "rotor_3"):
            prim = get_prim_at_path(f"{self._drone_root}/{link}")
            if prim and prim.IsValid() and prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                api = PhysxSchema.PhysxRigidBodyAPI(prim)
                damping[link] = [float(api.GetLinearDampingAttr().Get()),
                                 float(api.GetAngularDampingAttr().Get())]
        out["rigid_body_damping"] = damping
        return out

    @property
    def drone(self) -> Any:
        return self._drone

    @property
    def controller(self) -> Any:
        return self._controller

    @property
    def sim(self) -> Any:
        return self._sim

    @property
    def prim_path(self) -> str:
        assert self._drone_root is not None
        return self._drone_root

    @property
    def base_link_path(self) -> str:
        assert self._base_link_path is not None
        return self._base_link_path

    @property
    def ground_hit(self) -> bool:
        """True if the drone has descended below a safe altitude."""
        return float(self.get_state_np()[2]) < 0.03

    # -- shutdown ----------------------------------------------------------

    def close(self) -> None:
        """Close the simulation app exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._app is not None:
                self._app.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error while closing simulation app: %s", exc)

    def __enter__(self) -> "SimulationInterface":
        self.ensure_ready()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# IMU helpers (lazy imports; degrade gracefully if no IMU is available)
# ---------------------------------------------------------------------------


def _gravity_vector(g: Any) -> list[float]:
    """PhysicsContext.get_gravity() returns (direction, magnitude)."""
    direction, magnitude = g
    return [float(d) * float(magnitude) for d in direction]


def _initialize_imu(imu: Any) -> bool:
    try:
        imu.initialize()
    except TypeError:
        try:
            imu.initialize(None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("IMU initialize(None) failed: %s", exc)
            return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("IMU initialize() failed: %s", exc)
        return False
    if hasattr(imu, "resume"):
        try:
            imu.resume()
        except Exception:  # noqa: BLE001
            pass
    return True


def _create_imu_sensor(
    base_link_path: str,
    dt: float,
    frequency_hz: int,
    filter_size: int = 1,
    initialize: bool = True,
) -> Any | None:
    """Create and initialise a physics IMU on the rigid-body base link.

    Isaac Sim 4.1 uses ``omni.isaac.sensor.IMUSensor``; newer versions use
    ``isaacsim.sensors.physics.IMUSensor``. Both expose the same
    ``get_current_frame`` contract.
    """
    imu_cls = None
    for module in ("omni.isaac.sensor", "isaacsim.sensors.physics"):
        try:
            mod = __import__(module, fromlist=["IMUSensor"])
            imu_cls = getattr(mod, "IMUSensor")
            break
        except Exception:
            continue

    if imu_cls is None:
        logger.info("No IMUSensor class found in omni.isaac.sensor or isaacsim.sensors.physics")
        return None

    try:
        kwargs: dict[str, Any] = {
            "prim_path": f"{base_link_path}/Imu_Sensor",
            "name": "imu_sensor",
            "frequency": int(frequency_hz),
        }
        if filter_size > 1:
            kwargs["linear_acceleration_filter_size"] = int(filter_size)
            kwargs["angular_velocity_filter_size"] = int(filter_size)

        imu = imu_cls(**kwargs)
        if initialize and not _initialize_imu(imu):
            return None
        return imu

    except Exception as exc:
        logger.info("Could not create IMU sensor at %s: %s", base_link_path, exc)
        return None


def _read_imu_frame(imu: Any, read_gravity: bool) -> dict | None:
    """Read the latest frame, tolerating API differences across versions."""
    try:
        return imu.get_current_frame(read_gravity=read_gravity)
    except TypeError:
        # Older signature without read_gravity.
        try:
            return imu.get_current_frame()
        except Exception:
            return None
    except Exception:
        return None


def _extract_vec(frame: dict, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in frame:
            value = frame[key]
            # The physics IMU returns torch tensors on the sim device (cuda).
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float64).reshape(3)
    return None

__all__ = [
    "SimulationConfig",
    "SimulationInterface",
    "install_signal_handlers",
    "stop_requested",
    "reset_stop_flag",
]