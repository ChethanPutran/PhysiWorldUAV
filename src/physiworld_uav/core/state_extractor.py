"""State layout, quaternion utilities, and IMU fallback for OmniDrones/Isaac Sim.

**Simulator** state layout, as returned by OmniDrones ``drone.get_state()[..., :13]``
(verified from ``omni_drones/robots/drone/multirotor.py``: ``self.vel = self.vel_w``):

    index   field                         frame       units
    -----   ----------------------------  ----------  -----
    0:3     position                      world       m
    3:7     orientation (quaternion wxyz) body->world  --
    7:10    linear velocity               world       m/s
    10:13   angular velocity              **world**   rad/s

The thesis/model state uses the **body** angular velocity; convert with
``angular_velocity_body_torch`` / ``angular_velocity_body_np`` (plan §5-A11).
The model-ready layout (p | v | q | ω_B) is defined in ``data/schema.py``.

All quaternions are Hamilton convention, scalar-first (w, x, y, z), and
normalised (``|q| = 1``). Rotations map body-frame vectors to world-frame
vectors: ``v_world = R(q) @ v_body``.

This module is intentionally **dependency-light**: it imports ``numpy`` and
``torch`` at module level (both are base dependencies of the project) and
never imports ``omni_drones`` or ``omni.isaac``. That keeps it importable in
CI, in tests, and in any downstream analysis module that consumes logged
trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Layout constants — the single source of truth for state slicing
# ---------------------------------------------------------------------------

NUM_STATE: int = 13
POS: slice = slice(0, 3)
QUAT: slice = slice(3, 7)
VEL: slice = slice(7, 10)
ANGVEL: slice = slice(10, 13)

GRAVITY_MPS2: float = 9.81


# ---------------------------------------------------------------------------
# Torch-side helpers (used inside the simulation loop; must be batched)
# ---------------------------------------------------------------------------


def quat_rotate_torch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` (…, 3) by unit quaternions ``q`` (…, 4, wxyz): body → world."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def quat_rotate_inverse_torch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by the inverse of ``q``: world → body."""
    conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return quat_rotate_torch(conj, v)


def angular_velocity_body_torch(state: torch.Tensor) -> torch.Tensor:
    """ω_B = R_WBᵀ ω_W from a simulator-layout state (…, 13)."""
    return quat_rotate_inverse_torch(state[..., QUAT], state[..., ANGVEL])


def quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    conj = np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)
    return quat_rotate_np(conj, v)


def angular_velocity_body_np(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    return quat_rotate_inverse_np(state[..., QUAT], state[..., ANGVEL])


def split_state(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a batched state tensor ``(..., 13)`` into named components.

    Returns
    -------
    position, quaternion_wxyz, linear_velocity_world, angular_velocity_body
        Each has shape ``(..., k)`` with ``k`` in ``{3, 4, 3, 3}``.
    """
    if state.shape[-1] < NUM_STATE:
        raise ValueError(
            f"state last dim must be >= {NUM_STATE}, got {state.shape[-1]}"
        )
    return state[..., POS], state[..., QUAT], state[..., VEL], state[..., ANGVEL]


def quat_wxyz_to_matrix(q: torch.Tensor, *, eps: float = 1e-9) -> torch.Tensor:
    """Convert a (batched) quaternion ``wxyz`` to a rotation matrix.

    Parameters
    ----------
    q : torch.Tensor
        Shape ``(..., 4)``, scalar-first, Hamilton convention.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3, 3)`` mapping body vectors to world vectors.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"expected quaternion last dim 4, got {q.shape[-1]}")
    q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=eps)
    w, x, y, z = q.unbind(dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    row0 = torch.stack([m00, m01, m02], dim=-1)
    row1 = torch.stack([m10, m11, m12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def quat_continuity_torch(q: torch.Tensor) -> torch.Tensor:
    """Flip quaternion signs so that consecutive samples have ``dot >= 0``.

    Operates along the **last** axis (time). For batched sequences of shape
    ``(B, T, 4)``, use ``q.transpose(0, 1)`` before/after.
    """
    if q.shape[-1] != 4:
        raise ValueError("expected quaternion last dim 4")
    out = q.clone()
    dots = (out[..., 1:, :] * out[..., :-1, :]).sum(dim=-1)
    sign = torch.where(dots < 0.0, -1.0, 1.0)
    # Cumulative product along time ensures consistent global sign.
    sign = torch.cumprod(sign, dim=-1)
    out[..., 1:, :] = out[..., 1:, :] * sign.unsqueeze(-1)
    return out


def slerp_torch(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between ``q0`` and ``q1``.

    Parameters
    ----------
    q0, q1 : torch.Tensor, shape ``(..., 4)``
    t : torch.Tensor, shape broadcastable to ``q0[..., :1]``, in ``[0, 1]``.
    """
    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)

    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    small = sin_omega.abs() < 1e-6

    w0 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * omega) / sin_omega)
    w1 = torch.where(small, t, torch.sin(t * omega) / sin_omega)
    out = w0 * q0 + w1 * q1
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-9)


# ---------------------------------------------------------------------------
# Numpy-side helpers (used in logging / post-processing / dataset building)
# ---------------------------------------------------------------------------


def quat_wxyz_to_matrix_np(q: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q, axis=-1, keepdims=True).max(), eps)
    w, x, y, z = np.moveaxis(q, -1, 0)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    m[..., 0, 1] = 2.0 * (xy - wz)
    m[..., 0, 2] = 2.0 * (xz + wy)
    m[..., 1, 0] = 2.0 * (xy + wz)
    m[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    m[..., 1, 2] = 2.0 * (yz - wx)
    m[..., 2, 0] = 2.0 * (xz - wy)
    m[..., 2, 1] = 2.0 * (yz + wx)
    m[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return m


def quat_continuity_np(q: np.ndarray) -> np.ndarray:
    """Sign-continuous quaternion sequence along axis 0."""
    q = np.array(q, dtype=np.float64, copy=True)
    for i in range(1, q.shape[0]):
        if np.dot(q[i], q[i - 1]) < 0.0:
            q[i] = -q[i]
    return q


def quat_to_euler_np(q: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (ZYX) in radians. Shape ``(..., 3)``."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * np.pi / 2.0, np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1)


def compute_imu_from_state(
    state: np.ndarray,
    prev_velocity_world: np.ndarray,
    dt: float,
    *,
    gravity: float = GRAVITY_MPS2,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic IMU fallback derived from finite-differenced state.

    Accelerometer output is **specific force** in the body frame,
    ``f_b = R_bw @ (a_world - g_world)`` with ``g_world = [0, 0, -g]``.
    A stationary level drone therefore reads ``+g`` along body-z.

    Parameters
    ----------
    state : np.ndarray, shape ``(13,)``
        Current flat state (layout defined at module top).
    prev_velocity_world : np.ndarray, shape ``(3,)``
        Linear velocity at the previous step, world frame.
    dt : float
        Integration step in seconds; must be positive.

    Returns
    -------
    accel_body, gyro_body : np.ndarray, shape ``(3,)``
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.size < NUM_STATE:
        raise ValueError(f"state must have >= {NUM_STATE} elements, got {state.size}")

    v_world = state[VEL]
    a_world = (v_world - np.asarray(prev_velocity_world, dtype=np.float64)) / dt
    R_bw = quat_wxyz_to_matrix_np(state[QUAT])

    g_world = np.array([0.0, 0.0, -gravity])
    specific_force_world = a_world - g_world
    accel_body = R_bw.T @ specific_force_world

    gyro_body = R_bw.T @ state[ANGVEL]          # state ω is world-frame (see module doc)
    return accel_body, gyro_body


# ---------------------------------------------------------------------------
# Structured view for logging / dataset construction
# ---------------------------------------------------------------------------


@dataclass
class StateView:
    """Named view of one or more state samples (numpy, ready to log)."""

    position: np.ndarray            # (N, 3)
    orientation_wxyz: np.ndarray    # (N, 4)
    velocity_world: np.ndarray      # (N, 3)
    angular_velocity_body: np.ndarray  # (N, 3)
    time: np.ndarray | None = None  # (N,) optional

    @classmethod
    def from_array(cls, state: np.ndarray, time: np.ndarray | None = None) -> "StateView":
        arr = np.asarray(state, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[-1] < NUM_STATE:
            raise ValueError(f"state last dim must be >= {NUM_STATE}")
        return cls(
            position=arr[:, POS],
            orientation_wxyz=arr[:, QUAT],
            velocity_world=arr[:, VEL],
            angular_velocity_body=arr[:, ANGVEL],
            time=time,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "position": self.position,
            "orientation": self.orientation_wxyz,
            "velocity": self.velocity_world,
            "angular_velocity": self.angular_velocity_body,
        }
        if self.time is not None:
            out["t"] = self.time
        return out


__all__ = [
    "NUM_STATE",
    "POS",
    "QUAT",
    "VEL",
    "ANGVEL",
    "GRAVITY_MPS2",
    "StateView",
    "split_state",
    "quat_wxyz_to_matrix",
    "quat_continuity_torch",
    "slerp_torch",
    "quat_wxyz_to_matrix_np",
    "quat_continuity_np",
    "quat_to_euler_np",
    "compute_imu_from_state",
]