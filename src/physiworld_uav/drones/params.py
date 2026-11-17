"""Drone physical parameters (plan §2, §6.1-F2, §3-C4).

``DroneParams`` is the single in-code representation of a multirotor's
physical configuration. It is built from ``config/drone/<name>.yaml`` and is
generic (not Hummingbird-specific): adding a drone means adding a YAML file.

Conventions (verified against OmniDrones source and USD, audit §5):

* Body frame FLU, thrust along **+Z_B**.
* Rotor i sits in the body X–Y plane at
  ``r_i = (L_i cos θ_i, L_i sin θ_i, h_rotor)``; for the Hummingbird
  "+" layout rotor 0 is on +x_B.
* ``mass`` is the **total** vehicle mass (base link + rotor links).
* Rotor speed Ω ∈ [0, Ω_max] rad/s; thrust per rotor T = k_F Ω².
  OmniDrones' throttle satisfies throttle = Ω / Ω_max.
"""
from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

G = 9.81  # m/s², must equal the PhysX scene gravity magnitude

#: Keys compared with OmniDrones' own drone yaml (plan §2.2-S3).
_SHARED_ROTOR_KEYS = (
    "arm_lengths", "directions", "rotor_angles", "force_constants",
    "moment_constants", "max_rotation_velocities",
)


@dataclass(frozen=True)
class DroneParams:
    name: str
    mass: float                                   # kg, total
    inertia_diag: tuple[float, float, float]      # kg m², (Ixx, Iyy, Izz)
    inertia_offdiag: tuple[float, float, float]   # kg m², (Ixy, Ixz, Iyz)
    arm_lengths: tuple[float, ...]                # m
    rotor_angles: tuple[float, ...]               # rad, in body X–Y plane
    directions: tuple[float, ...]                 # spin sign (OmniDrones convention)
    force_constants: tuple[float, ...]            # k_F, N/(rad/s)²
    moment_constants: tuple[float, ...]           # k_M, N m/(rad/s)²
    max_rotor_speed: tuple[float, ...]            # rad/s
    rotor_height: float = 0.0                     # m
    rotor_link_mass: float = 0.0                  # kg per rotor link
    motor_alpha_nominal: float = 0.43             # OmniDrones per-step blend factor
    motor_alpha_reference_dt: float = 0.005       # s, dt at which alpha_nominal applies
    motor_time_constant: float | None = None      # s; None → derived from alpha_nominal
    drag_coef_omnidrones: float = 0.0             # OmniDrones' own drag (kept 0, A5)
    extra: Mapping[str, Any] = field(default_factory=dict)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "DroneParams":
        rot = cfg["rotor_configuration"]
        n = int(rot.get("num_rotors", len(rot["arm_lengths"])))
        for key in _SHARED_ROTOR_KEYS:
            if len(rot[key]) != n:
                raise ValueError(f"rotor_configuration.{key} must have {n} entries")
        inertia = cfg["inertia"]
        params = cls(
            name=str(cfg["name"]),
            mass=float(cfg["mass"]),
            inertia_diag=(float(inertia["xx"]), float(inertia["yy"]), float(inertia["zz"])),
            inertia_offdiag=(float(inertia.get("xy", 0.0)), float(inertia.get("xz", 0.0)),
                             float(inertia.get("yz", 0.0))),
            arm_lengths=tuple(float(x) for x in rot["arm_lengths"]),
            rotor_angles=tuple(float(x) for x in rot["rotor_angles"]),
            directions=tuple(float(x) for x in rot["directions"]),
            force_constants=tuple(float(x) for x in rot["force_constants"]),
            moment_constants=tuple(float(x) for x in rot["moment_constants"]),
            max_rotor_speed=tuple(float(x) for x in rot["max_rotation_velocities"]),
            rotor_height=float(cfg.get("rotor_height", 0.0)),
            rotor_link_mass=float(cfg.get("rotor_link_mass", 0.0)),
            motor_alpha_nominal=float(cfg.get("motor_alpha_nominal", 0.43)),
            motor_alpha_reference_dt=float(cfg.get("motor_alpha_reference_dt", 0.005)),
            motor_time_constant=(float(cfg["motor_time_constant"])
                                 if cfg.get("motor_time_constant") is not None else None),
            drag_coef_omnidrones=float(cfg.get("drag_coef", 0.0)),
        )
        params.validate()
        return params

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DroneParams":
        from physiworld_uav.core.config_loader import load_yaml
        return cls.from_dict(load_yaml(path))

    # -- validation ------------------------------------------------------
    def validate(self) -> None:
        if self.mass <= 0:
            raise ValueError(f"mass must be > 0, got {self.mass}")
        Ixx, Iyy, Izz = self.inertia_diag
        if min(Ixx, Iyy, Izz) <= 0:
            raise ValueError(f"principal inertia must be > 0, got {self.inertia_diag}")
        if not (Ixx + Iyy >= Izz and Ixx + Izz >= Iyy and Iyy + Izz >= Ixx):
            raise ValueError(f"inertia violates the triangle inequality: {self.inertia_diag}")
        if min(self.force_constants) <= 0 or min(self.moment_constants) <= 0:
            raise ValueError("rotor force/moment constants must be > 0")
        if not 0.0 < self.motor_alpha_nominal <= 1.0:
            raise ValueError("motor_alpha_nominal must be in (0, 1]")
        if self.thrust_to_weight < 1.0:
            raise ValueError(f"thrust-to-weight {self.thrust_to_weight:.2f} < 1: cannot hover")

    # -- derived quantities ----------------------------------------------
    @property
    def num_rotors(self) -> int:
        return len(self.arm_lengths)

    @property
    def inertia_matrix(self) -> np.ndarray:
        Ixx, Iyy, Izz = self.inertia_diag
        Ixy, Ixz, Iyz = self.inertia_offdiag
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

    @property
    def rotor_positions_B(self) -> np.ndarray:
        """(N, 3) rotor hub positions in the body frame (X–Y plane, +Z offset)."""
        return np.array([
            [L * math.cos(a), L * math.sin(a), self.rotor_height]
            for L, a in zip(self.arm_lengths, self.rotor_angles)
        ])

    @property
    def thrust_axis_B(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    @property
    def max_thrust_per_rotor(self) -> np.ndarray:
        return np.asarray(self.force_constants) * np.asarray(self.max_rotor_speed) ** 2

    @property
    def max_thrust(self) -> float:
        return float(self.max_thrust_per_rotor.sum())

    @property
    def weight(self) -> float:
        return self.mass * G

    @property
    def thrust_to_weight(self) -> float:
        return self.max_thrust / self.weight

    @property
    def hover_rotor_speed(self) -> np.ndarray:
        """Rotor speeds (rad/s) for which Σ k_F Ω² = m g with equal Ω."""
        kf = np.asarray(self.force_constants)
        return np.sqrt(self.weight / kf.sum()) * np.ones(self.num_rotors)

    @property
    def hover_throttle(self) -> np.ndarray:
        return self.hover_rotor_speed / np.asarray(self.max_rotor_speed)

    @property
    def hover_command(self) -> np.ndarray:
        """OmniDrones rotor command in [-1, 1] that holds hover in steady state.

        OmniDrones: target_throttle = sqrt((cmd + 1)/2) ⇒ cmd = 2 throttle² − 1.
        """
        return 2.0 * self.hover_throttle ** 2 - 1.0

    @property
    def effective_motor_time_constant(self) -> float:
        if self.motor_time_constant is not None:
            return self.motor_time_constant
        return -self.motor_alpha_reference_dt / math.log(1.0 - self.motor_alpha_nominal)

    def motor_alpha(self, dt: float) -> float:
        """Per-step blend factor reproducing the first-order lag τ_m at step ``dt``."""
        return 1.0 - math.exp(-dt / self.effective_motor_time_constant)

    # -- transforms ------------------------------------------------------
    def scaled(self, *, mass: float | None = None,
               inertia_diag: tuple[float, float, float] | None = None,
               force_scale: float = 1.0, moment_scale: float = 1.0,
               motor_time_constant: float | None = None) -> "DroneParams":
        """Return a copy with domain-randomised values (validated)."""
        new = replace(
            self,
            mass=self.mass if mass is None else float(mass),
            inertia_diag=self.inertia_diag if inertia_diag is None else tuple(map(float, inertia_diag)),
            force_constants=tuple(k * force_scale for k in self.force_constants),
            moment_constants=tuple(k * moment_scale for k in self.moment_constants),
            motor_time_constant=(self.effective_motor_time_constant
                                 if motor_time_constant is None else float(motor_time_constant)),
        )
        new.validate()
        return new

    def to_omnidrones_params(self) -> dict[str, Any]:
        """OmniDrones ``uav_params`` dict (controllers take mass/inertia from here)."""
        Ixx, Iyy, Izz = self.inertia_diag
        Ixy, Ixz, Iyz = self.inertia_offdiag
        return {
            "name": self.name,
            "mass": self.mass,
            "inertia": {"xx": Ixx, "yy": Iyy, "zz": Izz, "xy": Ixy, "xz": Ixz, "yz": Iyz},
            "rotor_configuration": {
                "num_rotors": self.num_rotors,
                "arm_lengths": list(self.arm_lengths),
                "rotor_angles": list(self.rotor_angles),
                "directions": list(self.directions),
                "force_constants": list(self.force_constants),
                "moment_constants": list(self.moment_constants),
                "max_rotation_velocities": list(self.max_rotor_speed),
            },
        }

    def summary(self) -> dict[str, Any]:
        """Values recorded in episode metadata."""
        return {
            "name": self.name,
            "mass_kg": self.mass,
            "inertia_diag_kgm2": list(self.inertia_diag),
            "inertia_offdiag_kgm2": list(self.inertia_offdiag),
            "force_constants": list(self.force_constants),
            "moment_constants": list(self.moment_constants),
            "max_rotor_speed_radps": list(self.max_rotor_speed),
            "motor_time_constant_s": self.effective_motor_time_constant,
            "max_thrust_N": self.max_thrust,
            "thrust_to_weight": self.thrust_to_weight,
        }


# ---------------------------------------------------------------------------
# Cross-check with OmniDrones (plan §2.2-S3)
# ---------------------------------------------------------------------------


def omnidrones_yaml_path(name: str) -> Path | None:
    spec = importlib.util.find_spec("omni_drones")
    if spec is not None and spec.origin:
        base = Path(spec.origin).parent
    else:
        base = Path(__file__).resolve().parents[3] / "OmniDrones" / "omni_drones"
    path = base / "robots" / "assets" / "usd" / f"{name}.yaml"
    return path if path.exists() else None


def check_against_omnidrones(params: DroneParams, *, rtol: float = 1e-9) -> list[str]:
    """Return mismatches between our YAML and OmniDrones' for shared keys.

    Compares rotor geometry/constants and inertia. Mass is deliberately not
    compared exactly: OmniDrones' yaml lists the total mass (0.716) like ours,
    but domain-randomised params will differ; call this on the *nominal* params.
    """
    path = omnidrones_yaml_path(params.name)
    if path is None:
        return [f"OmniDrones yaml for {params.name!r} not found"]
    with path.open() as f:
        od = yaml.safe_load(f)
    issues: list[str] = []
    ours = params.to_omnidrones_params()
    for key in _SHARED_ROTOR_KEYS:
        a = np.asarray(ours["rotor_configuration"][key], dtype=float)
        b = np.asarray(od["rotor_configuration"][key], dtype=float)
        if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=1e-12):
            issues.append(f"rotor_configuration.{key}: ours={a.tolist()} omnidrones={b.tolist()}")
    for k in ("xx", "yy", "zz", "xy", "xz", "yz"):
        if not math.isclose(ours["inertia"][k], float(od["inertia"].get(k, 0.0)),
                            rel_tol=rtol, abs_tol=1e-12):
            issues.append(f"inertia.{k}: ours={ours['inertia'][k]} omnidrones={od['inertia'].get(k)}")
    if not math.isclose(params.mass, float(od["mass"]), rel_tol=rtol):
        issues.append(f"mass: ours={params.mass} omnidrones={od['mass']}")
    return issues


def load_drone(path: str | Path = "config/drone/hummingbird.yaml", *, strict: bool = True) -> DroneParams:
    """Load params and enforce agreement with OmniDrones (raises if ``strict``)."""
    params = DroneParams.from_yaml(path)
    issues = check_against_omnidrones(params)
    if issues and strict:
        raise ValueError("drone config disagrees with OmniDrones:\n  " + "\n  ".join(issues))
    return params
