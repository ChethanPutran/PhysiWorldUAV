"""Quaternion utilities, Hamilton convention, wxyz order, body→world (matches the dataset).

Conventions
-----------
* Quaternions are stored scalar-first, ``q = [w, x, y, z]``, on the last dim.
* Hamilton product (``ij = k``). A unit quaternion ``q`` represents the rotation
  ``R(q)`` that maps body-frame vectors to world-frame vectors: ``v_W = R(q) v_B``.
* ``q`` and ``-q`` encode the same rotation. Functions that compare or convert
  rotations (``geodesic_angle``, ``to_rotation_vector``) handle this sign ambiguity.
* Angular velocity ``ω_B`` is expressed in the body frame, which is why the
  kinematics use right-multiplication ``q ⊗ [0, ω_B]``.

All functions are batched over leading dimensions and differentiable.
"""
from __future__ import annotations

import torch


def normalize(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Scale quaternions to unit norm.

    Args:
        q: Quaternions, shape (..., 4).
        eps: Lower bound on the norm, to avoid division by zero for a
            (near-)zero input. Such an input is returned scaled by 1/eps
            rather than raising or producing NaN.

    Returns:
        Unit quaternions, shape (..., 4).
    """
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate ``[w, -x, -y, -z]``.

    For a unit quaternion this is the inverse rotation (world → body instead of
    body → world).

    Args:
        q: Quaternions, shape (..., 4).

    Returns:
        Conjugates, shape (..., 4).
    """
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a ⊗ b``.

    Not commutative. If ``a`` rotates body → world and ``b`` is a rotation in
    the body frame, ``a ⊗ b`` applies ``b`` first, then ``a``. Does not
    normalize; the product of two unit quaternions is unit up to rounding.

    Args:
        a: Left quaternions, shape (..., 4), wxyz.
        b: Right quaternions, shape (..., 4), wxyz. Must broadcast with ``a``.

    Returns:
        Product, shape (..., 4), wxyz.
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """R(q) with v_W = R v_B, for a unit quaternion.

    The input is normalized internally, so slightly off-norm quaternions still
    give a proper rotation matrix. Column ``j`` of ``R`` is body axis ``j``
    expressed in the world frame (e.g. ``R[..., :, 2]`` is body z in world).

    Args:
        q: Quaternions, shape (..., 4), wxyz.

    Returns:
        Rotation matrices, shape (..., 3, 3).
    """
    w, x, y, z = normalize(q).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def omega_matrix_derivative(q: torch.Tensor, omega_b: torch.Tensor) -> torch.Tensor:
    """q̇ = ½ q ⊗ [0, ω_B].

    Quaternion kinematics for a body-frame angular velocity (right
    multiplication). For a world-frame ω_W the form would be ``½ [0, ω_W] ⊗ q``.

    ``q`` is used as given (not normalized here), and the result does not
    preserve ‖q‖ = 1, so the caller must renormalize after integrating.

    Args:
        q: Orientation, shape (..., 4), wxyz, body → world.
        omega_b: Angular velocity in body axes [rad/s], shape (..., 3).

    Returns:
        Quaternion time derivative, shape (..., 4).
    """
    zero = torch.zeros_like(omega_b[..., :1])
    return 0.5 * mul(q, torch.cat([zero, omega_b], dim=-1))


def from_rotation_vector(rv: torch.Tensor) -> torch.Tensor:
    """Exponential map: rotation vector (rad) → unit quaternion; stable near zero.

    The rotation vector's direction is the rotation axis and its norm is the
    angle. Below an angle of 1e-6 rad, ``sin(θ/2)/θ`` is replaced by its Taylor
    expansion ``½ - θ²/48`` to avoid 0/0, and the clamp keeps gradients finite.

    Args:
        rv: Rotation vectors [rad], shape (..., 3).

    Returns:
        Unit quaternions, shape (..., 4), wxyz, with w ≥ 0 for angles in [0, π].
    """
    angle = rv.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    small = angle < 1e-6
    k = torch.where(small, 0.5 - angle * angle / 48.0, torch.sin(half) / angle.clamp_min(1e-12))
    return torch.cat([torch.cos(half), k * rv], dim=-1)


def geodesic_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """d_q(q1, q2) = 2·atan2(‖Im(q1 ⊗ q2*)‖, |Re(q1 ⊗ q2*)|) in radians.

    Sign-invariant (q and −q are the same rotation) and numerically stable at 0,
    unlike 2·arccos|⟨q1,q2⟩|. Equals the geodesic angle between R(q1) and R(q2).

    Both inputs are assumed to be unit quaternions.

    Args:
        q1: Quaternions, shape (..., 4), wxyz.
        q2: Quaternions, shape (..., 4), wxyz. Must broadcast with ``q1``.

    Returns:
        Rotation angle between the two orientations, shape (...,), in [0, π].
    """
    d = mul(q1, conj(q2))
    return 2.0 * torch.atan2(d[..., 1:].norm(dim=-1), d[..., 0].abs())


def to_rotation_vector(q: torch.Tensor) -> torch.Tensor:
    """Inverse of ``from_rotation_vector`` (shortest rotation, angle in [0, π]).

    The input is normalized and flipped to w ≥ 0 first, so ``q`` and ``-q`` give
    the same output. For a near-zero rotation (‖Im q‖ < 1e-8) the scale factor
    ``angle / ‖Im q‖`` is replaced by its limit, 2.

    Args:
        q: Quaternions, shape (..., 4), wxyz.

    Returns:
        Rotation vectors [rad], shape (..., 3), with norm in [0, π].
    """
    q = normalize(q)
    sign = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    q = q * sign
    im = q[..., 1:]
    n = im.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(n, q[..., :1])
    scale = torch.where(n < 1e-8, 2.0 * torch.ones_like(n), angle / n.clamp_min(1e-12))
    return im * scale