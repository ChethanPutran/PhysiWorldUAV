"""Assemble the enabled aerodynamic models for one vehicle (plan §6.1, D5).

``AeroForceSystem.from_config(cfg, nominal, domain_factors)`` reads a
``config/forces/<name>.yaml`` dict. ``compute(...)`` returns per-component and
total wrenches plus per-model applicability-envelope violations for one
physics step. Gravity and rotor thrust/reaction torque are **not** here: they
are applied by PhysX / OmniDrones (force accounting, docs/models/force_accounting.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor

from physiworld_uav.core.state_extractor import ANGVEL, POS, QUAT, VEL, quat_rotate_inverse_torch, quat_wxyz_to_matrix
from physiworld_uav.drones.params import DroneParams

from . import models as M
from .registry import REGISTRY

COMPONENTS = ("rotor_drag", "body_drag", "ground_effect", "aero_torque")


@dataclass
class WrenchResult:
    force_W: dict[str, Tensor]                 # component -> (N, 3)
    torque_B: dict[str, Tensor]                # component -> (N, 3)
    total_force_W: Tensor
    total_torque_B: Tensor
    total_torque_W: Tensor
    violations: dict[str, Tensor] = field(default_factory=dict)   # component -> (N,) bool


@dataclass
class AeroForceSystem:
    name: str
    validated: bool
    rotor_drag: M.RotorDragParams | None
    body_drag: M.BodyDragParams | None
    ground_effect: M.GroundEffectParams | None
    aero_torque: M.AeroTorqueParams | None
    r_B: Tensor
    k_f: Tensor
    max_rotor_speed: Tensor
    rotor_radius: float
    envelopes: dict[str, dict[str, float]]

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(c for c in COMPONENTS if getattr(self, c) is not None)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], drone: DroneParams, nominal: DroneParams,
                    factors: Mapping[str, float] | None = None) -> "AeroForceSystem":
        factors = dict(factors or {})
        rho = float(cfg.get("air_density", 1.225))
        R = float(cfg.get("rotor_radius_m", 0.10))
        rd = cfg.get("rotor_drag") or {}
        bd = cfg.get("body_drag") or {}
        ge = cfg.get("ground_effect") or {}
        at = cfg.get("aero_torque") or {}
        rotor = None
        if rd.get("enabled", False):
            if "k_rd" in rd:
                k_rd = float(rd["k_rd"])
            else:
                # mass-normalised coefficient d [1/s] at nominal hover:  k_rd = d m₀ / ΣΩ_hover,₀
                k_rd = float(rd["d_hover_per_s"]) * nominal.mass / float(nominal.hover_rotor_speed.sum())
            rotor = M.RotorDragParams(k_rd=k_rd, scale=float(factors.get("rotor_drag_scale", 1.0)))
        body = None
        if bd.get("enabled", False):
            cd = bd["cd"]
            cd = tuple(float(c) for c in cd) if isinstance(cd, (list, tuple)) else float(cd)
            body = M.BodyDragParams(cd=cd, area_m2=tuple(float(a) for a in bd["area_m2"]),
                                    air_density=rho,
                                    cop_offset_B=tuple(float(x) for x in bd.get("cop_offset_B", (0, 0, 0))),
                                    scale=float(factors.get("body_drag_scale", 1.0)))
        ground = None
        if ge.get("enabled", False):
            ground = M.GroundEffectParams(rotor_radius=R,
                                          min_height_over_radius=float(ge.get("min_height_over_radius", 0.5)))
        torque = None
        if at.get("enabled", False):
            torque = M.AeroTorqueParams(k1=tuple(float(x) for x in at["k1"]), k2=float(at.get("k2", 0.0)))
        env = {k: dict(REGISTRY[k].envelope) for k in COMPONENTS}
        for k, v in (cfg.get("applicability") or {}).items():
            env[k].update({kk: float(vv) for kk, vv in v.items()})
        return cls(name=str(cfg.get("name", "unnamed")), validated=bool(cfg.get("validated", False)),
                   rotor_drag=rotor, body_drag=body, ground_effect=ground, aero_torque=torque,
                   r_B=torch.as_tensor(drone.rotor_positions_B, dtype=torch.float32),
                   k_f=torch.as_tensor(drone.force_constants, dtype=torch.float32),
                   max_rotor_speed=torch.as_tensor(drone.max_rotor_speed, dtype=torch.float32),
                   rotor_radius=R, envelopes=env)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "validated": self.validated, "enabled": list(self.enabled)}
        for c in self.enabled:
            out[c] = {k: v for k, v in getattr(self, c).__dict__.items()}
        out["rotor_radius_m"] = self.rotor_radius
        return out

    def rotor_positions_W(self, state: Tensor) -> Tensor:
        R_WB = quat_wxyz_to_matrix(state[:, QUAT])
        return state[:, POS].unsqueeze(1) + torch.einsum("nij,kj->nki", R_WB, self.r_B.to(state.device))

    def compute(self, state: Tensor, omega: Tensor, wind_cm: Tensor, wind_rot: Tensor) -> WrenchResult:
        """``state``: (N, 13) simulator layout (ω world). Returns body/world wrenches."""
        dev = state.device
        R_WB = quat_wxyz_to_matrix(state[:, QUAT])
        v_W = state[:, VEL]
        w_B = quat_rotate_inverse_torch(state[:, QUAT], state[:, ANGVEL])
        r_B = self.r_B.to(dev)
        fB: dict[str, Tensor] = {}
        tB: dict[str, Tensor] = {}
        viol: dict[str, Tensor] = {}
        if self.rotor_drag is not None:
            fB["rotor_drag"], tB["rotor_drag"] = M.rotor_drag(self.rotor_drag, R_WB, v_W, w_B, omega, wind_rot, r_B)
            v_rel = torch.einsum("nji,nkj->nki", R_WB, v_W.unsqueeze(1) - wind_rot)
            mu = torch.linalg.norm(v_rel[..., :2], dim=-1) / (omega.clamp_min(1.0) * self.rotor_radius)
            e = self.envelopes["rotor_drag"]
            viol["rotor_drag"] = ((mu > e["max_advance_ratio"]).any(-1)
                                  | (omega < e["min_rotor_speed_fraction"] * self.max_rotor_speed.to(dev)).any(-1))
        if self.body_drag is not None:
            fB["body_drag"], tB["body_drag"] = M.body_drag(self.body_drag, R_WB, v_W, wind_cm)
            viol["body_drag"] = torch.linalg.norm(v_W - wind_cm, dim=-1) > self.envelopes["body_drag"]["max_airspeed_mps"]
        if self.ground_effect is not None:
            fB["ground_effect"], tB["ground_effect"] = M.ground_effect(
                self.ground_effect, R_WB, state[:, POS], omega, self.k_f.to(dev), r_B)
            hub_z = self.rotor_positions_W(state)[..., 2]
            viol["ground_effect"] = (hub_z < self.envelopes["ground_effect"]["min_height_over_radius"]
                                     * self.rotor_radius).any(-1)
        if self.aero_torque is not None:
            fB["aero_torque"], tB["aero_torque"] = M.aero_torque(self.aero_torque, w_B)
            viol["aero_torque"] = torch.linalg.norm(w_B, dim=-1) > self.envelopes["aero_torque"]["max_body_rate_radps"]
        n = state.shape[0]
        fW = {k: torch.einsum("nij,nj->ni", R_WB, v) for k, v in fB.items()}
        zero = torch.zeros(n, 3, device=dev)
        tot_f = sum(fW.values(), zero)
        tot_tB = sum(tB.values(), zero)
        return WrenchResult(force_W=fW, torque_B=tB, total_force_W=tot_f, total_torque_B=tot_tB,
                            total_torque_W=torch.einsum("nij,nj->ni", R_WB, tot_tB), violations=viol)
