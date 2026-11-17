"""WindField — composition of wind components (design doc §7)."""
from __future__ import annotations

import numpy as np

from physiworld_uav.core.seeds import child_seed

from .base import WindComponent, WindSample


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class WindField:
    """Evaluates all enabled components and returns the total + breakdown."""

    def __init__(
        self,
        components: dict[str, WindComponent],
        wind_scale: float = 1.0,
        temporal_scale: float = 1.0,
        rotate_z_rad: float = 0.0,
    ) -> None:
        if wind_scale < 0:
            raise ValueError("wind_scale must be >= 0")
        if temporal_scale <= 0:
            raise ValueError("temporal_scale must be > 0")
        self.components = dict(components)
        self.wind_scale = float(wind_scale)
        self.temporal_scale = float(temporal_scale)
        # Counterfactual "same realisation, rotated" (plan §6.2-W9): the whole
        # field, every component, is rotated about world z.
        self.rotate_z_rad = float(rotate_z_rad)
        self._R = _rot_z(self.rotate_z_rad) if self.rotate_z_rad != 0.0 else None

    def evaluate(
        self,
        position_W: np.ndarray,
        time_s: float,
        dt: float | None = None,
    ) -> WindSample:
        pos = np.asarray(position_W, dtype=np.float64).reshape(3)

        eff_t = time_s * self.temporal_scale
        eff_dt = dt * self.temporal_scale if dt is not None else None

        by_component: dict[str, np.ndarray] = {}
        for name, comp in self.components.items():
            # Temporal scaling applies to every component (plan §6.2-W5): stateful
            # components advance by the scaled step, which is equivalent to dividing
            # their time constants by ``temporal_scale`` (OU τ, Dryden L/V, event rate).
            v = comp.contribution(pos, eff_t, eff_dt)
            v = np.asarray(v, dtype=np.float64) * self.wind_scale
            if self._R is not None:
                v = self._R @ v
            by_component[name] = v

        total = np.zeros(3)
        for v in by_component.values():
            total = total + v
        speed = float(np.linalg.norm(total))
        direction = total / speed if speed > 1e-9 else np.zeros(3)

        return WindSample(
            position_W=pos.copy(),
            time_s=float(time_s),
            total_W=total,
            by_component=by_component,
            speed_mps=speed,
            direction_unit_W=direction,
        )

    def reset_all(self, seed: int | None = None) -> None:
        """Reset every component with its *own* child seed (plan §6.2-W4).

        The child seed is derived from ``seed`` and the component's stable name,
        so two components never share a random stream, and a sibling episode
        with the same seed and the same component names reproduces the same
        realisation (required for counterfactual pairs).
        """
        for name, comp in self.components.items():
            comp.reset(None if seed is None else child_seed(int(seed), f"wind_component:{name}"))