"""
Nominal rigid-body quadrotor dynamics ``f_phys`` for the 13-dim model state
(thesis B1, B3).

Returns the time derivative ds/dt of the state under textbook rigid-body
physics with nominal parameters only. In the hybrid model, a learned residual
corrects what this function omits:

    ds/dt ≈ f_phys(s, u) + f_residual(s, u)

State and control
-----------------
s = [p_W (3), v_W (3), q_wxyz (4), ω_B (3)]                      (13,)

    p_W     position of the center of mass in the world frame        [m]
    v_W     velocity in the world frame                              [m/s]
    q_wxyz  unit quaternion, scalar-first (Hamilton), rotating
            body → world, so R(q) maps B-frame vectors to W
    ω_B     angular velocity of the body relative to the world,
            expressed in body axes                                   [rad/s]

u = [Ω_1, …, Ω_4]                                                (4,)

    Rotor speeds [rad/s]. By decision D3 these are the interval-mean
    speeds over the control step, i.e. already post motor lag.

Equations
---------
    ṗ = v
    v̇ = g_W + R(q) · (ΣT_i / m) · ẑ_B,        T_i = k_F,i · Ω_i²
    q̇ = ½ q ⊗ [0, ω_B]
    J ω̇ = τ − ω × (J ω)

    τ = Σ_i r_i × (T_i ẑ_B)  +  yaw_sign · Σ_i dir_i · k_M,i · Ω_i² · ẑ_B

Frame conventions
-----------------
All quantities in the rotational equations are in the body frame B:
ω = ω_B, τ = τ_B, r_i = rotor positions relative to the CoM, and J = J_B,
the constant inertia matrix about the CoM in body axes (not necessarily
diagonal). R(q) is the only W↔B conversion and acts on the thrust vector.
g_W is the gravity vector in the world frame (e.g. [0, 0, -9.81] for z-up).

Parameters (nominal, fixed)
---------------------------
    m           mass [kg]
    J           inertia matrix, body frame [kg·m²]
    k_F,i       per-rotor thrust coefficient [N/(rad/s)²]
    k_M,i       per-rotor reaction-torque coefficient [N·m/(rad/s)²]
    r_i         rotor positions in B, relative to the CoM [m]
    dir_i       rotor spin direction, ±1
    yaw_sign    ±1, sets the sign convention of positive yaw

Notes
-----
* q̇ does not preserve ‖q‖ = 1. The integrator or caller must renormalize q
  after each step.
* Thrust uses the mean rotor speed, so T_i = k_F,i · Ω̄_i² differs slightly
  from the true mean of k_F,i · Ω_i(t)². This small bias is left to the
  residual.

Not modeled (left to the learned residual)
------------------------------------------
Wind, aerodynamic drag, ground effect, motor lag (absorbed into u via D3),
rotor gyroscopic and inertial torques, blade flapping, and parameter error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from . import quat
from physiworld_uav.drones.params import DroneParams, G

STATE_DIM, ACTION_DIM = 13, 4
P, V, Q, W = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13)


class QuadrotorPhysics(nn.Module):
    """Nominal rigid-body quadrotor model ``f_phys`` for the 13-dim state.

    Implements the textbook dynamics with nominal parameters only: no wind,
    drag, ground effect or motor lag. A learned residual can be added through
    the ``extra`` argument of :meth:`derivative`.

    State layout (last dim, 13), indexed by the slices ``P, V, Q, W``::

        s = [p_W (3), v_W (3), q_wxyz (4), ω_B (3)]

    ``q`` is a scalar-first unit quaternion rotating body → world. ``ω_B`` is
    the body-frame angular velocity [rad/s].

    Control layout (last dim, 4): rotor speeds ``Ω_i`` [rad/s], the
    interval-mean speeds over one control step (decision D3, so already post
    motor lag).

    Conventions:
        * World frame is z-up; gravity is ``[0, 0, -g]``.
        * Rotor positions lie in the body x-y plane (no z offset).
        * Torque, ω and J are all in the body frame; ``R(q)`` is the only
          body → world conversion and is applied to the thrust vector.

    All parameters are stored as buffers (moved with ``.to(device)``, not
    trained).

    Args:
        params: Nominal drone parameters (mass, arm lengths, rotor angles,
            thrust/moment constants, spin directions, full 3x3 inertia).
        yaw_torque_sign: ±1 sign convention for the reaction yaw torque.
            The default (-1) was verified against the pilot data
            (tests/cpu/test_models.py).

    Buffers:
        rx, ry (4,): rotor x/y positions in body axes [m].
        kf (4,): thrust coefficients, T_i = kf_i Ω_i² [N/(rad/s)²].
        km (4,): reaction-torque coefficients [N·m/(rad/s)²].
        dirs (4,): rotor spin directions, ±1.
        J, J_inv (3, 3): body-frame inertia matrix about the CoM and its
            inverse [kg·m²].
    """

    def __init__(self, params: DroneParams, yaw_torque_sign: float = -1.0) -> None:
        super().__init__()
        self.mass = float(params.mass)
        r = np.array([[L * np.cos(a), L * np.sin(a)] for L, a in zip(params.arm_lengths, params.rotor_angles)])
        self.register_buffer("rx", torch.tensor(r[:, 0], dtype=torch.float32))
        self.register_buffer("ry", torch.tensor(r[:, 1], dtype=torch.float32))
        self.register_buffer("kf", torch.tensor(params.force_constants, dtype=torch.float32))
        self.register_buffer("km", torch.tensor(params.moment_constants, dtype=torch.float32))
        self.register_buffer("dirs", torch.tensor(params.directions, dtype=torch.float32))
        J = torch.tensor(params.inertia_matrix, dtype=torch.float32)
        self.register_buffer("J", J)
        self.register_buffer("J_inv", torch.linalg.inv(J))
        self.yaw_torque_sign = float(yaw_torque_sign)   # verified against the pilot data (tests/cpu/test_models.py)
        self.g = float(G)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, **kw) -> "QuadrotorPhysics":
        """Build the model from a drone YAML file.

        Args:
            path: Path to the drone config. If None, uses
                ``config/drone/hummingbird.yaml`` under the repository root
                (three directories above this file).
            **kw: Forwarded to the constructor (e.g. ``yaw_torque_sign``).

        Returns:
            A configured :class:`QuadrotorPhysics`.
        """
        path = Path(path) if path else Path(__file__).resolve().parents[3] / "config" / "drone" / "hummingbird.yaml"
        return cls(DroneParams.from_yaml(path), **kw)

    def wrench_B(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Total thrust and body torque produced by the rotors.

        With T_i = kf_i Ω_i² acting along body +z at position (rx_i, ry_i):

            τ_x =  Σ ry_i T_i                      (roll)
            τ_y = -Σ rx_i T_i                      (pitch)
            τ_z = yaw_sign · Σ dir_i km_i Ω_i²     (yaw, reaction torque)

        Args:
            u: Rotor speeds Ω_i [rad/s], shape (..., 4).

        Returns:
            thrust: Total thrust ΣT_i along body +z [N], shape (...,).
            torque: Body-frame torque [N·m], shape (..., 3).
        """
        T = self.kf * u * u
        torque = torch.stack([
            (self.ry * T).sum(-1),
            -(self.rx * T).sum(-1),
            self.yaw_torque_sign * (self.dirs * self.km * u * u).sum(-1),
        ], dim=-1)
        return T.sum(-1), torque

    def derivative(self, s: torch.Tensor, u: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """Time derivative ṡ of the 13-dim state, optionally plus a residual.

        Nominal dynamics::

            ṗ = v
            v̇ = R(q) (ΣT_i / m) ẑ_B - g ẑ_W
            q̇ = ½ q ⊗ [0, ω]
            ω̇ = J⁻¹ (τ - ω × Jω)

        The quaternion is normalised internally before use, so slightly
        un-normalised inputs are tolerated.

        Args:
            s: State, shape (..., 13).
            u: Rotor speeds [rad/s], shape (..., 4).
            extra: Optional additive tangent-space term, shape (..., 12),
                laid out as ``[Δṗ (3), Δv̇ (3), δ (3), Δω̇ (3)]``:

                * ``Δṗ``, ``Δv̇``, ``Δω̇`` are added to the position,
                  velocity and angular-acceleration derivatives.
                * ``δ`` is an attitude-rate correction. It is added to ω
                  **only inside the quaternion kinematics**
                  (q̇ = ½ q ⊗ [0, ω + δ]); the gyroscopic term still uses
                  the unmodified ω.

                The 12-dim tangent form avoids a residual on the 4-dim
                quaternion, which would leave the unit sphere.

        Returns:
            ṡ, shape (..., 13), in the same layout as the state. The
            quaternion derivative does not preserve ‖q‖ = 1; callers must
            renormalise after integrating (see :meth:`project`).
        """
        v, q, w = s[..., V], quat.normalize(s[..., Q]), s[..., W]
        thrust, torque = self.wrench_B(u)
        R = quat.to_rotation_matrix(q)
        z_b = R[..., :, 2]                                          # body z in world
        acc = z_b * (thrust / self.mass).unsqueeze(-1)
        acc = acc - torch.tensor([0.0, 0.0, self.g], device=s.device, dtype=s.dtype)
        Jw = w @ self.J.T
        wdot = (torque - torch.cross(w, Jw, dim=-1)) @ self.J_inv.T
        pdot = v
        w_kin = w
        if extra is not None:
            pdot = pdot + extra[..., 0:3]
            acc = acc + extra[..., 3:6]
            w_kin = w + extra[..., 6:9]
            wdot = wdot + extra[..., 9:12]
        qdot = quat.omega_matrix_derivative(q, w_kin)
        return torch.cat([pdot, acc, qdot, wdot], dim=-1)

    def project(self, s: torch.Tensor) -> torch.Tensor:
        """Project a state back onto the valid manifold.

        Renormalises the quaternion to unit length and leaves position,
        velocity and angular velocity unchanged.

        Args:
            s: State, shape (..., 13).

        Returns:
            State with a unit quaternion, shape (..., 13).
        """
        return torch.cat([s[..., P], s[..., V], quat.normalize(s[..., Q]), s[..., W]], dim=-1)


def rk4_step(deriv: Callable[[torch.Tensor], torch.Tensor], s: torch.Tensor, dt: float,
             project: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Advance the state by one classical RK4 step with the control held constant.

    The control ``u`` is assumed constant over the step (zero-order hold) and
    is captured by the closure ``deriv``. The quaternion is renormalised on
    the intermediate stage inputs (k2, k3, k4) and on the final result, so the
    state stays on the unit-quaternion manifold.

    Args:
        deriv: Function ``s -> ṡ`` for fixed ``u``, e.g.
            ``lambda x: model.derivative(x, u)``.
        s: Current state, shape (..., 13).
        dt: Step length [s].
        project: Function mapping a state back onto the valid manifold, e.g.
            ``model.project``.

    Returns:
        State after ``dt``, shape (..., 13), with a unit quaternion.
    """
    k1 = deriv(s)
    k2 = deriv(project(s + 0.5 * dt * k1))
    k3 = deriv(project(s + 0.5 * dt * k2))
    k4 = deriv(project(s + dt * k3))
    return project(s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))
