from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, List, Optional, Union


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_list(x):
    """Convert OmegaConf ListConfig / tuple / ndarray / scalar to plain list."""
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return list(x)
    return [x]


def _is_range(v) -> bool:
    """True iff v looks like [min, max]."""
    return isinstance(v, (list, tuple, np.ndarray)) and len(v) == 2 \
        and all(isinstance(x, (int, float)) for x in v) \
        and float(v[0]) <= float(v[1])


def _sample(rng: np.random.Generator, v):
    """If v is [min, max], sample uniformly; else return v as float."""
    if _is_range(v):
        return float(rng.uniform(float(v[0]), float(v[1])))
    return float(v)


def _resolve_direction(spec_dir, direction_table, rng) -> np.ndarray:
    """
    direction spec can be:
      - a 3-vector [x, y, z]
      - a name in direction_table (e.g. "north")
      - a list of names/vectors  → sample one
      - "from_list" placeholder with a parallel field
      - "current_heading" string (resolved by caller, not here)
    """
    if spec_dir is None:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    if isinstance(spec_dir, str):
        if spec_dir == "current_heading":
            raise ValueError("current_heading must be resolved by caller")
        if spec_dir in direction_table:
            return np.array(direction_table[spec_dir], dtype=np.float32).reshape(3)
        raise KeyError(f"Unknown direction name: {spec_dir!r}")

    dirs = _to_list(spec_dir)
    if dirs is None or len(dirs) == 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # If the outer list has multiple entries, pick one
    if len(dirs) >= 1 and not np.isscalar(dirs[0]) and not isinstance(dirs[0], str):
        # Single 3-vector or a list of 3-vectors?
        try:
            if len(dirs) == 3 and all(np.isscalar(d) for d in dirs):
                return np.asarray(dirs, dtype=np.float32).reshape(3)
        except Exception:
            pass
        chosen = dirs[rng.integers(len(dirs))]
        return _resolve_direction(chosen, direction_table, rng)

    # List of names
    chosen = dirs[rng.integers(len(dirs))]
    return _resolve_direction(chosen, direction_table, rng)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.zeros(3, dtype=np.float32)


def _quat_to_roll_pitch(q: np.ndarray) -> tuple[float, float]:
    """q = (w, x, y, z) → (roll, pitch) in radians."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    return roll, pitch

def _euler_to_quat(roll, pitch, yaw):
    """Return (w, x, y, z) quaternion for intrinsic XYZ Euler angles."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)

def _quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _enforce_quaternion_continuity(q: np.ndarray) -> np.ndarray:
    """
    Input: Nx4 unit quaternions in [w, x, y, z] order.
    Output: sign-consistent quaternion sequence.
    """
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def _set_physx_force(base_link_prim, force_xyz):
    from pxr import PhysxSchema, Gf
    rb = PhysxSchema.PhysxRigidBodyAPI(base_link_prim)
    rb.GetForceAttr().Set(Gf.Vec3f(
        float(force_xyz[0]), float(force_xyz[1]), float(force_xyz[2])
    ))