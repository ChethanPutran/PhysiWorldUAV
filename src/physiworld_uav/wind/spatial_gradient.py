"""Spatial wind gradient — design doc §7 ``G(p - p₀)`` term."""
from __future__ import annotations

import numpy as np


class SpatialGradient:
    """Linear spatial wind gradient ``u_grad(p) = G @ (p - p0)``, units 1/s."""

    name = "gradient"
    is_stateful = False

    def __init__(
        self,
        G: np.ndarray,
        p0: np.ndarray | None = None,
    ) -> None:
        G = np.asarray(G, dtype=np.float64)
        if G.shape != (3, 3):
            raise ValueError(f"G must be 3x3, got shape {G.shape}")
        self.G = G
        self.p0 = np.zeros(3) if p0 is None else np.asarray(p0, dtype=np.float64).reshape(3)

    def contribution(self, position_W, time_s, dt=None) -> np.ndarray:
        return self.G @ (np.asarray(position_W, dtype=np.float64) - self.p0)

    def reset(self, seed=None) -> None:
        pass