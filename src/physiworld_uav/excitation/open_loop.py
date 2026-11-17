"""Open-loop excitation programs (plan §6.3-T11; prompt §6B).

Three **physically distinct** abstractions, recorded as ``open_loop_kind``:

1. ``rotor_command`` — pure open loop at the actuator. Signals are defined in
   mixer space (Δ collective thrust [N], Δ body torques [N m]) around the hover
   trim and converted to per-rotor thrusts through the exact allocation of the
   drone's geometry, then to OmniDrones rotor commands
   ``cmd_i = 2 T_i / T_max,i − 1``. No feedback of any kind.
2. ``collective_rate`` — collective thrust [N] + body-rate targets [rad/s] fed
   to OmniDrones' ``RateController``. The inner **rate loop is closed**: this
   is "rate-stabilised open loop", not pure open loop.
3. ``body_wrench`` — rotors held at trim, an external body wrench is applied
   through the force injector. Not a physical UAV input; identification-only
   (excluded from model-ready data unless selected). Requires the Run 4 injector.

All programs start from hover equilibrium unless the scenario overrides it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from physiworld_uav.drones.params import DroneParams

from . import signals as S

KINDS = ("rotor_command", "collective_rate", "body_wrench")
CHANNELS = {
    "rotor_command": ("collective", "tau_x", "tau_y", "tau_z"),
    "collective_rate": ("collective", "rate_x", "rate_y", "rate_z"),
    "body_wrench": ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"),
}


def allocation_matrix(drone: DroneParams) -> np.ndarray:
    """A such that [τx, τy, τz, F]ᵀ = A · T (per-rotor thrusts), body frame.

    τ = Σ r_i × (T_i e₃) gives τx = y_i T_i, τy = −x_i T_i; the yaw reaction
    torque per rotor is −dir_i (k_M/k_F) T_i (OmniDrones sign convention, same
    rows as ``omni_drones.controllers.compute_parameters``).
    """
    r = drone.rotor_positions_B
    kf = np.asarray(drone.force_constants)
    km = np.asarray(drone.moment_constants)
    d = np.asarray(drone.directions)
    return np.stack([r[:, 1], -r[:, 0], -d * km / kf, np.ones(drone.num_rotors)])


@dataclass(frozen=True)
class OpenLoopProgram:
    kind: str
    channels: dict[str, dict[str, Any]]           # channel -> concrete signal
    trim_thrust_N: float                          # collective trim (= m g at hover)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"open_loop kind must be one of {KINDS}")
        bad = set(self.channels) - set(CHANNELS[self.kind])
        if bad:
            raise ValueError(f"channels {sorted(bad)} invalid for {self.kind}")

    def _v(self, ch: str, t: float) -> float:
        sig = self.channels.get(ch)
        return 0.0 if sig is None else S.evaluate(sig, t)

    # -- rotor_command -----------------------------------------------------
    def rotor_thrusts(self, drone: DroneParams, t: float) -> np.ndarray:
        """Per-rotor thrust targets [N] (unclipped)."""
        if self.kind != "rotor_command":
            raise TypeError("rotor_thrusts() only for rotor_command programs")
        wrench = np.array([self._v("tau_x", t), self._v("tau_y", t), self._v("tau_z", t),
                           self.trim_thrust_N + self._v("collective", t)])
        return np.linalg.solve(allocation_matrix(drone), wrench)

    def rotor_command(self, drone: DroneParams, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(applied cmd ∈ [−1, 1], raw cmd) — the raw value records saturation."""
        raw = 2.0 * self.rotor_thrusts(drone, t) / drone.max_thrust_per_rotor - 1.0
        return np.clip(raw, -1.0, 1.0), raw

    # -- collective_rate ---------------------------------------------------
    def collective_rate(self, t: float) -> tuple[float, np.ndarray]:
        if self.kind != "collective_rate":
            raise TypeError("collective_rate() only for collective_rate programs")
        return (self.trim_thrust_N + self._v("collective", t),
                np.array([self._v("rate_x", t), self._v("rate_y", t), self._v("rate_z", t)]))

    # -- body_wrench -------------------------------------------------------
    def body_wrench(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if self.kind != "body_wrench":
            raise TypeError("body_wrench() only for body_wrench programs")
        return (np.array([self._v(f"force_{a}", t) for a in "xyz"]),
                np.array([self._v(f"torque_{a}", t) for a in "xyz"]))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "channels": self.channels, "trim_thrust_N": self.trim_thrust_N,
                "extra": self.extra}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OpenLoopProgram":
        return cls(kind=d["kind"], channels=dict(d["channels"]),
                   trim_thrust_N=float(d["trim_thrust_N"]), extra=dict(d.get("extra", {})))


def make_program(kind: str, templates: Mapping[str, Mapping[str, Any]], drone: DroneParams,
                 rng: np.random.Generator, duration: float) -> OpenLoopProgram:
    return OpenLoopProgram(kind=kind, channels=S.sample_channels(templates, rng, duration),
                           trim_thrust_N=drone.weight)
