"""Control laws on top of OmniDrones' controllers (plan §6.3-T1/T2/T3, Q2).

No controller is reimplemented here: position tracking uses OmniDrones'
``LeePositionController`` (SO(3), yaw-aware, tilt-projected thrust,
acceleration feed-forward), attitude tracking uses ``AttitudeController``,
collective + body-rate commands use ``RateController``.

Controllers are built from the **true domain parameters** (decision Q2:
``controller_param_policy = "true"``); ``"nominal"`` is available for
ablations. The policy is recorded in episode metadata.

Frame handling (verified from OmniDrones source):
* simulator state layout: p | q_wxyz | v_W | ω_W  (13)
* Lee / Rate controllers convert ω_W → ω_B internally (``quat_rotate_inverse``)
* AttitudeController uses the state's angular velocity **as body rate**, so it
  receives a state whose last three entries are ω_B.

All functions are batched over a leading environment dimension (N, ·).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from physiworld_uav.core.state_extractor import quat_rotate_inverse_torch
from physiworld_uav.drones.params import G, DroneParams
from physiworld_uav.trajectories.library import RefPoint

MODES = ("position", "attitude", "rate", "rotor_command", "replay")
POLICIES = ("true", "nominal")


@dataclass(frozen=True)
class AttitudeHoldGains:
    """Slow, yaw-aware outer loop that keeps attitude manoeuvres near their start point."""
    kp_xy: float = 0.8
    kd_xy: float = 1.2
    kp_z: float = 2.0
    kd_z: float = 1.5
    max_tilt_rad: float = 0.35

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "AttitudeHoldGains":
        return cls(**dict(d or {}))


@dataclass
class ControlOutput:
    """Per-env control output for one control step."""
    cmd: torch.Tensor                    # (N, R) applied rotor command ∈ [-1, 1]
    cmd_raw: torch.Tensor                # (N, R) controller output before clamping
    collective_thrust: torch.Tensor      # (N,) N, commanded collective thrust (nan if n/a)
    body_rate_cmd: torch.Tensor          # (N, 3) rad/s (nan if n/a)
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


class ControlLaw:
    """Mode-specific control law for N environments sharing one drone model.

    Separate ``ControlLaw`` instances are used for different domains (the
    mixer depends on inertia and rotor constants).
    """

    def __init__(self, mode: str, drone: DroneParams, device: str | torch.device = "cpu",
                 *, gains: Mapping[str, Any] | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        from omni_drones.controllers import AttitudeController, LeePositionController, RateController

        self.mode = mode
        self.drone = drone
        self.device = torch.device(device)
        params = drone.to_omnidrones_params()
        gains = dict(gains or {})
        self.hold = AttitudeHoldGains.from_dict(gains.pop("attitude_hold", None))
        self._lee = self._att = self._rate = None
        if mode == "position":
            self._lee = LeePositionController(G, params).to(self.device)
            for key, attr in (("position_gain", "pos_gain"), ("velocity_gain", "vel_gain")):
                if key in gains:
                    getattr(self._lee, attr).data = torch.as_tensor(gains.pop(key), dtype=torch.float32,
                                                                    device=self.device)
        elif mode == "attitude":
            self._att = AttitudeController(G, params).to(self.device)
        elif mode == "rate":
            self._rate = RateController(G, params).to(self.device)
        if gains:
            raise ValueError(f"unsupported gain overrides for mode {mode!r}: {sorted(gains)}")
        for c in (self._lee, self._att, self._rate):
            if c is not None:
                c.requires_grad_(False)     # Rate/Attitude controllers leave grads enabled
        self._max_thrust = torch.as_tensor(drone.max_thrust_per_rotor, dtype=torch.float32,
                                           device=self.device)

    # -- helpers --------------------------------------------------------------
    def _out(self, raw: torch.Tensor, thrust=None, rates=None, **extras) -> ControlOutput:
        n = raw.shape[0]
        nan = torch.full((n,), float("nan"), device=raw.device)
        return ControlOutput(
            cmd=raw.clamp(-1.0, 1.0), cmd_raw=raw,
            collective_thrust=nan if thrust is None else thrust.reshape(n),
            body_rate_cmd=torch.full((n, 3), float("nan"), device=raw.device) if rates is None
            else rates.reshape(n, 3),
            extras=extras,
        )

    @staticmethod
    def _stack(points: Sequence[RefPoint], attr: str, device) -> torch.Tensor:
        return torch.as_tensor(np.stack([np.asarray(getattr(p, attr), dtype=np.float32) for p in points]),
                               device=device)

    # -- modes ------------------------------------------------------------------
    def position(self, state: torch.Tensor, refs: Sequence[RefPoint]) -> ControlOutput:
        """``state``: (N, 13) simulator layout (ω world). ``refs``: N position RefPoints."""
        dev = state.device
        raw = self._lee.compute(
            state,
            target_pos=self._stack(refs, "position", dev),
            target_vel=self._stack(refs, "velocity", dev),
            target_acc=self._stack(refs, "acceleration", dev),
            target_yaw=torch.as_tensor([[p.yaw] for p in refs], dtype=torch.float32, device=dev),
        )
        return self._out(raw)

    def attitude(self, state: torch.Tensor, refs: Sequence[RefPoint],
                 hold_position: torch.Tensor | None) -> ControlOutput:
        """Attitude manoeuvre with an optional yaw-aware position/altitude hold.

        ``hold_position``: (N, 3) point to hold (None → only altitude rate damping).
        Collective thrust is tilt-compensated: T = m (g + a_z) / (cos φ cos θ) (plan T2).
        """
        dev = state.device
        n = state.shape[0]
        p, q, v, w_W = state[:, 0:3], state[:, 3:7], state[:, 7:10], state[:, 10:13]
        roll = torch.as_tensor([[r.roll] for r in refs], dtype=torch.float32, device=dev)
        pitch = torch.as_tensor([[r.pitch] for r in refs], dtype=torch.float32, device=dev)
        yaw_rate = torch.as_tensor([[r.yaw_rate or 0.0] for r in refs], dtype=torch.float32, device=dev)
        h = self.hold
        az = -h.kd_z * v[:, 2]
        if hold_position is not None:
            e = hold_position - p
            az = az + h.kp_z * e[:, 2]
            a_xy = h.kp_xy * e[:, :2] - h.kd_xy * v[:, :2]
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                              1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            c, s = torch.cos(yaw), torch.sin(yaw)
            ax_b = c * a_xy[:, 0] + s * a_xy[:, 1]          # rotate world → yaw frame
            ay_b = -s * a_xy[:, 0] + c * a_xy[:, 1]
            pitch = pitch + (ax_b / G).clamp(-h.max_tilt_rad, h.max_tilt_rad).unsqueeze(-1)
            roll = roll + (-ay_b / G).clamp(-h.max_tilt_rad, h.max_tilt_rad).unsqueeze(-1)
        tilt_cos = (torch.cos(roll) * torch.cos(pitch)).squeeze(-1).clamp_min(0.5)
        thrust = (self.drone.mass * (G + az) / tilt_cos).clamp(
            0.1 * self.drone.max_thrust, 0.9 * self.drone.max_thrust).unsqueeze(-1)
        state_b = torch.cat([state[:, :10], quat_rotate_inverse_torch(q, w_W)], dim=-1)
        raw = self._att(state_b, target_thrust=thrust, target_yaw_rate=yaw_rate,
                        target_roll=roll, target_pitch=pitch)
        return self._out(raw, thrust=thrust, roll_cmd=roll.squeeze(-1), pitch_cmd=pitch.squeeze(-1))

    def rate(self, state: torch.Tensor, collective_N: torch.Tensor, body_rate: torch.Tensor) -> ControlOutput:
        raw = self._rate(state, target_rate=body_rate.reshape(-1, 3),
                         target_thrust=collective_N.reshape(-1, 1))
        return self._out(raw, thrust=collective_N, rates=body_rate)

    def rotor_command(self, raw_cmd: torch.Tensor) -> ControlOutput:
        return self._out(raw_cmd)


def controller_params(nominal: DroneParams, domain_drone: DroneParams, policy: str) -> DroneParams:
    """Parameters the controller is built from (decision Q2)."""
    if policy not in POLICIES:
        raise ValueError(f"controller_param_policy must be one of {POLICIES}")
    return domain_drone if policy == "true" else nominal
