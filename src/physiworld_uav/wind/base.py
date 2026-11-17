"""Protocol and shared data types for the wind subsystem (design doc §7)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# Directions table — used when a spec provides a name instead of a vector.
DEFAULT_DIRECTIONS: dict[str, np.ndarray] = {
    "north": np.array([1.0, 0.0, 0.0]),
    "south": np.array([-1.0, 0.0, 0.0]),
    "east":  np.array([0.0, 1.0, 0.0]),
    "west":  np.array([0.0, -1.0, 0.0]),
    "up":    np.array([0.0, 0.0, 1.0]),
    "down":  np.array([0.0, 0.0, -1.0]),
}


def resolve_direction(spec, directions: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Resolve ``spec`` to a unit 3-vector.

    Accepts a name (``"north"``), a 3-vector, or a list of those (one is chosen
    at random by the caller — this helper only accepts concrete choices).
    """
    table = directions or DEFAULT_DIRECTIONS
    if isinstance(spec, str):
        if spec not in table:
            raise KeyError(f"unknown direction name: {spec!r}")
        v = table[spec]
    else:
        v = np.asarray(spec, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError(f"direction {spec!r} has zero magnitude")
    return v / n


@dataclass
class WindSample:
    """One evaluation of the wind field, with per-component breakdown."""

    position_W: np.ndarray            # (3,) m
    time_s: float
    total_W: np.ndarray               # (3,) m/s
    by_component: dict[str, np.ndarray]  # name -> (3,) m/s
    speed_mps: float
    direction_unit_W: np.ndarray      # (3,) unit vector; zeros if speed < 1e-9

    def __add__(self, other: "WindSample") -> "WindSample":
        total = self.total_W + other.total_W
        speed = float(np.linalg.norm(total))
        by = dict(self.by_component)
        for k, v in other.by_component.items():
            by[k] = by.get(k, 0.0) + v
        return WindSample(
            position_W=self.position_W,
            time_s=self.time_s,
            total_W=total,
            by_component=by,
            speed_mps=speed,
            direction_unit_W=(total / speed if speed > 1e-9 else np.zeros(3)),
        )


@runtime_checkable
class WindComponent(Protocol):
    """Every wind contribution implements this protocol."""

    name: str
    is_stateful: bool

    def contribution(
        self, position_W: np.ndarray, time_s: float, dt: float | None = None
    ) -> np.ndarray:
        """Velocity contribution at ``position_W`` and ``time_s`` (world frame, m/s)."""
        ...

    def reset(self, seed: int | None = None) -> None:
        """Clear/reseed internal state for a new episode."""
        ...