"""Aerodynamic force/torque models (plan §6.1-F5…F8, decision D5). Batched torch.

All inputs use explicit frames; N = environments, R = rotors.

* ``R_WB``  (N, 3, 3) body → world rotation
* ``v_W``   (N, 3)    vehicle velocity, world
* ``w_B``   (N, 3)    body angular velocity
* ``omega`` (N, R)    rotor speeds Ω_i [rad/s] (post motor lag)
* ``wind_cm`` (N, 3) / ``wind_rot`` (N, R, 3)  wind velocity at the CoM / rotor hubs, world
* ``r_B``   (R, 3)    rotor hub positions, body (X–Y plane, +Z offset)

Every model returns ``(force_B, torque_B)`` in the **body** frame, applied at the
base-link origin (torques about it). Equations, assumptions and validity
envelopes are documented in ``docs/models/aerodynamics.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

E3 = torch.tensor([0.0, 0.0, 1.0])


def _to_body(R_WB: Tensor, v: Tensor) -> Tensor:
    """R_WBᵀ v for v of shape (N, 3) or (N, K, 3)."""
    if v.dim() == 2:
        return torch.einsum("nji,nj->ni", R_WB, v)
    return torch.einsum("nji,nkj->nki", R_WB, v)


# ---------------------------------------------------------------------------
# Rotor-induced drag (blade flapping + induced drag), linear in airspeed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RotorDragParams:
    k_rd: float              # N s / (m · rad/s): F_i = −k_rd Ω_i v⊥_i
    scale: float = 1.0       # domain randomisation factor


def rotor_drag(p: RotorDragParams, R_WB: Tensor, v_W: Tensor, w_B: Tensor, omega: Tensor,
               wind_rot: Tensor, r_B: Tensor) -> tuple[Tensor, Tensor]:
    """F_i,B = −k_rd Ω_i P⊥ v_rel,i,B,  v_rel,i,B = R_WBᵀ(v_W − w_i) + ω_B × r_i,  P⊥ = diag(1, 1, 0).

    Linear in the in-plane relative air velocity and proportional to rotor speed
    (Kai et al. 2017; Faessler et al. 2018: at hover Σ F = −m D v⊥ with
    D = diag(d, d, 0), d = k_rd ΣΩ_hover / m). τ_B = Σ r_i × F_i.
    """
    v_rel = _to_body(R_WB, v_W.unsqueeze(1) - wind_rot)                       # (N, R, 3)
    v_rel = v_rel + torch.linalg.cross(w_B.unsqueeze(1).expand_as(v_rel), r_B.expand_as(v_rel), dim=-1)
    v_perp = v_rel * torch.tensor([1.0, 1.0, 0.0], device=v_rel.device)
    F_i = -(p.k_rd * p.scale) * omega.unsqueeze(-1) * v_perp                  # (N, R, 3)
    tau = torch.linalg.cross(r_B.expand_as(F_i), F_i, dim=-1).sum(dim=1)
    return F_i.sum(dim=1), tau


# ---------------------------------------------------------------------------
# Body parasitic drag, quadratic, orientation-dependent projected area
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyDragParams:
    cd: float | tuple[float, float, float]        # drag coefficient (bluff body); scalar or per-axis (C_D,x, C_D,y, C_D,z)
    area_m2: tuple[float, float, float]           # projected areas normal to x_B, y_B, z_B
    air_density: float = 1.225                    # kg/m³
    cop_offset_B: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0


def body_drag(p: BodyDragParams, R_WB: Tensor, v_W: Tensor, wind_cm: Tensor) -> tuple[Tensor, Tensor]:
    """F_B = −½ ρ (C_D A)(û) ‖v_rel‖ v_rel,  (C_D A)(û) = Σ_j C_D,j A_j |û_j|,  v_rel = R_WBᵀ(v_W − w).

    Direction-consistent (force anti-parallel to the relative velocity) and
    orientation-dependent through the projected area of the body along the
    flow direction. τ_B = r_cop × F_B.
    """
    v_rel = _to_body(R_WB, v_W - wind_cm)                                       # (N, 3)
    speed = torch.linalg.norm(v_rel, dim=-1, keepdim=True)
    u = v_rel / speed.clamp_min(1e-9)
    cd = p.cd if isinstance(p.cd, (tuple, list)) else (p.cd,) * 3
    CdA = (u.abs() * torch.tensor(cd, device=v_rel.device) * torch.tensor(p.area_m2, device=v_rel.device)).sum(-1, keepdim=True)
    F = -0.5 * p.air_density * p.scale * CdA * speed * v_rel
    r = torch.tensor(p.cop_offset_B, device=F.device).expand_as(F)
    return F, torch.linalg.cross(r, F, dim=-1)


# ---------------------------------------------------------------------------
# Ground effect (Cheeseman–Bennett), per rotor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundEffectParams:
    rotor_radius: float
    min_height_over_radius: float = 0.5           # below this the factor is clamped


def ground_effect(p: GroundEffectParams, R_WB: Tensor, p_W: Tensor, omega: Tensor, k_f: Tensor,
                  r_B: Tensor) -> tuple[Tensor, Tensor]:
    """ΔT_i = (k_GE,i − 1) k_F Ω_i² along +z_B at rotor i, k_GE = 1/(1 − (R/4h_i)²).

    h_i is the rotor hub height above the ground plane z = 0, clamped to
    ``min_height_over_radius``·R. Derived for constant power; used here at fixed
    rotor speed (approximation, documented).
    """
    hub_W = p_W.unsqueeze(1) + torch.einsum("nij,kj->nki", R_WB, r_B)          # (N, R, 3)
    h = hub_W[..., 2].clamp_min(p.min_height_over_radius * p.rotor_radius)
    k = 1.0 / (1.0 - (p.rotor_radius / (4.0 * h)) ** 2)
    dT = (k - 1.0) * k_f * omega ** 2                                           # (N, R)
    F_i = dT.unsqueeze(-1) * E3.to(dT.device)                                   # body z
    tau = torch.linalg.cross(r_B.expand_as(F_i), F_i, dim=-1).sum(dim=1)
    return F_i.sum(dim=1), tau


# ---------------------------------------------------------------------------
# Rotational aerodynamic damping (ablation only; D5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AeroTorqueParams:
    k1: tuple[float, float, float]                # N m s / rad
    k2: float = 0.0                               # N m s² / rad² (isotropic → frame-invariant)


def aero_torque(p: AeroTorqueParams, w_B: Tensor) -> tuple[Tensor, Tensor]:
    """τ_B = −K₁ ω_B − k₂ ‖ω_B‖ ω_B (scalar k₂ keeps the quadratic term frame-invariant)."""
    K1 = torch.tensor(p.k1, device=w_B.device)
    tau = -K1 * w_B - p.k2 * torch.linalg.norm(w_B, dim=-1, keepdim=True) * w_B
    return torch.zeros_like(w_B), tau
