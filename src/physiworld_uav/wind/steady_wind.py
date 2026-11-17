"""Steady (uniform) wind — design doc §7 ``u₀`` term."""
from __future__ import annotations

import numpy as np

from .base import resolve_direction


class SteadyWind:
    """Constant uniform wind vector in the world frame."""

    name = "steady"
    is_stateful = False

    def __init__(
        self,
        speed_mps: float,
        direction,
        directions_table: dict[str, np.ndarray] | None = None,
    ) -> None:
        if speed_mps < 0:
            raise ValueError(f"steady wind speed must be >= 0, got {speed_mps}")
        self.speed_mps = float(speed_mps)
        self.direction_unit = resolve_direction(direction, directions_table)
        self._velocity = self.speed_mps * self.direction_unit

    def contribution(self, position_W, time_s, dt=None) -> np.ndarray:
        return self._velocity

    def reset(self, seed=None) -> None:
        pass