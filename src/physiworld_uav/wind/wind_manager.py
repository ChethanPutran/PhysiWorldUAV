"""WindManager — per-episode wind configuration and evaluation.

Responsibilities
----------------
1. Translate a YAML wind spec into a ``WindField`` of concrete components.
2. Sample any randomized parameters (magnitudes, directions, gust timings)
   **once per episode**, using the manager's seeded RNG.
3. Apply field-level modifiers (``wind_scale``, ``temporal_scale``, ``rotate_z_rad``).
   (``direction_override`` was removed: it rectified zero-mean components;
   directions are now chosen at spec level, plan §6.2-W6.)
4. Provide the documented API::

       wind_state = manager.get_wind_state(position=..., time=...)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import DEFAULT_DIRECTIONS, WindSample, resolve_direction
from .gust_generator import (
    CosineGust,
    OUGust,
    RandomGustSequence,
    SinusoidalGust,
    StepGust,
)
from .spatial_gradient import SpatialGradient
from .steady_wind import SteadyWind
from .turbulence import DrydenTurbulence
from .wind_field import WindField


class WindManager:
    def __init__(
        self,
        spec: dict[str, Any],
        seed: int | None = None,
        directions_table: dict[str, np.ndarray] | None = None,
    ) -> None:
        self._spec = dict(spec or {})
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._directions = dict(directions_table or DEFAULT_DIRECTIONS)
        self._field: WindField | None = None
        if self._spec.get("ood", {}).get("direction_override") is not None:
            raise ValueError("ood.direction_override was removed; set directions in the spec "
                             "or use ood.rotate_z_rad (plan §6.2-W6)")

    # ------------------------------------------------------------------ #
    def configure(self, spec_override: dict | None = None,
                  seed: int | None = None) -> "WindManager":
        """Build (or rebuild) the internal field. Idempotent per episode."""
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
        spec = self._spec if spec_override is None else {**self._spec, **spec_override}
        self._field = self._build_field(spec)
        self._field.reset_all(self._seed)
        return self

    def spatial_offsets(self, delta_W: np.ndarray) -> np.ndarray:
        """Wind difference at points displaced by ``delta_W`` (…, 3) from the last
        evaluation point. Only the spatial-gradient component is position dependent;
        stateful components (OU, Dryden, events) are spatially uniform and are
        evaluated once per instant (so their state advances exactly once)."""
        delta_W = np.asarray(delta_W, dtype=np.float64)
        grad = self._field.components.get("gradient") if self._field is not None else None
        if grad is None:
            return np.zeros_like(delta_W)
        out = delta_W @ np.asarray(grad.G, dtype=np.float64).T * self._field.wind_scale
        if self._field._R is not None:
            out = out @ self._field._R.T
        return out

    def get_wind_state(
        self,
        position: np.ndarray,
        time: float,
        dt: float | None = None,
    ) -> WindSample:
        if self._field is None:
            self.configure()
        sample = self._field.evaluate(np.asarray(position), time, dt)
        return sample

    # ------------------------------------------------------------------ #
    def _build_field(self, spec: dict) -> WindField:
        components: dict[str, Any] = {}

        if (steady := spec.get("steady")) and steady.get("enabled", True):
            speed = self._draw_scalar(steady.get("speed_mps", 0.0))
            direction = self._draw_direction(steady.get("direction", "north"))
            components["steady"] = SteadyWind(speed, direction, self._directions)

        if (grad := spec.get("gradient")) and grad.get("enabled", False):
            components["gradient"] = SpatialGradient(
                G=np.asarray(grad["G"], dtype=np.float64),
                p0=np.asarray(grad.get("p0", [0.0, 0.0, 0.0])),
            )

        gust_specs = spec.get("gust") or []
        if isinstance(gust_specs, dict):
            gust_specs = [gust_specs]
        for i, g in enumerate(gust_specs):
            if not g.get("enabled", True):
                continue
            components[f"gust_{i}"] = self._build_gust(g)

        turb = spec.get("turbulence")
        if turb and turb.get("enabled", False):
            steady_W = components["steady"].contribution(None, 0.0) if "steady" in components else np.zeros(3)
            components["turbulence"] = self._build_turbulence(turb, np.asarray(steady_W, dtype=float))

        ood = spec.get("ood", {})
        return WindField(
            components=components,
            wind_scale=float(ood.get("wind_scale", 1.0)),
            temporal_scale=float(ood.get("temporal_scale", 1.0)),
            rotate_z_rad=float(ood.get("rotate_z_rad", 0.0)),
        )

    # ------------------------------------------------------------------ #
    def _build_gust(self, g: dict):
        gtype = g.get("type", "cosine")
        if gtype == "cosine":
            return CosineGust(
                peak_mps=self._draw_scalar(g["peak_mps"]),
                start_time_s=self._draw_scalar(g.get("start_time_s", 0.0)),
                rise_time_s=self._draw_scalar(g.get("rise_time_s", 0.5)),
                direction=self._draw_direction(g.get("direction", "north")),
                directions_table=self._directions,
            )
        if gtype == "step":
            return StepGust(
                before_mps=self._draw_scalar(g.get("before_mps", 0.0)),
                after_mps=self._draw_scalar(g["after_mps"]),
                step_time_s=self._draw_scalar(g.get("step_time_s", 0.0)),
                rise_time_s=self._draw_scalar(g.get("rise_time_s", 0.0)),
                direction=self._draw_direction(g.get("direction", "north")),
                directions_table=self._directions,
            )
        if gtype == "sinusoidal":
            return SinusoidalGust(
                amplitude_mps=self._draw_scalar(g["amplitude_mps"]),
                frequency_hz=self._draw_scalar(g["frequency_hz"]),
                mean_mps=self._draw_scalar(g.get("mean_mps", 0.0)),
                direction=self._draw_direction(g.get("direction", "north")),
                directions_table=self._directions,
            )
        if gtype == "ou":
            return OUGust(
                sigma_mps=self._draw_scalar(g.get("sigma_mps", 1.0)),
                tau_s=self._draw_scalar(g.get("tau_s", 2.0)),
                vertical_ratio=self._draw_scalar(g.get("vertical_ratio", 0.5)),
                mean_mps=g.get("mean_mps", (0.0, 0.0, 0.0)),
                seed=self._draw_seed(),
            )
        if gtype == "random_events":
            return RandomGustSequence(
                rate_hz=self._draw_scalar(g.get("rate_hz", 0.2)),
                peak_range_mps=tuple(g.get("peak_range_mps", (2.0, 5.0))),
                rise_range_s=tuple(g.get("rise_range_s", (0.3, 1.2))),
                duration_s=self._draw_scalar(g.get("duration_s", 30.0))
                * float(self._spec.get("ood", {}).get("temporal_scale", 1.0)),
                directions_table=self._directions,
                seed=self._draw_seed(),
            )
        raise ValueError(f"unknown gust type: {gtype!r}")

    def _build_turbulence(self, t: dict, steady_W: np.ndarray) -> DrydenTurbulence:
        """Dryden turbulence. Either explicit ``intensities_mps`` + ``length_scales_m``, or
        MIL-F-8785C low altitude from ``altitude_m`` + ``w20_mps``. The frozen-turbulence
        speed defaults to the steady (mean) horizontal wind speed (Run 4, W3)."""
        model = t.get("model", "dryden")
        if model != "dryden":
            raise ValueError(f"turbulence model {model!r} not implemented; only 'dryden' is supported")
        if "w20_mps" in t:
            from .turbulence import low_altitude
            sig, L = low_altitude(self._draw_scalar(t.get("altitude_m", 3.05)), self._draw_scalar(t["w20_mps"]))
            sig = tuple(float(x) * float(t.get("intensity_scale", 1.0)) for x in sig)
        else:
            sig = tuple(float(x) for x in t["intensities_mps"])
            L = tuple(float(x) for x in t["length_scales_m"])
        mean_h = float(np.linalg.norm(steady_W[:2]))
        V = self._draw_scalar(t["airspeed_mps"]) if "airspeed_mps" in t else mean_h
        return DrydenTurbulence(
            airspeed_mps=V, intensities_mps=sig, length_scales_m=L,
            mean_W=np.asarray(t.get("mean_W", (0.0, 0.0, 0.0))),
            mean_direction_W=steady_W if mean_h > 1e-9 else None,
            min_airspeed_mps=float(t.get("min_airspeed_mps", 1.0)),
            seed=self._draw_seed(),
        )

    # ------------------------------------------------------------------ #
    def _draw_scalar(self, spec) -> float:
        """Draw one scalar from a constant or a ``[min, max]`` range."""
        if isinstance(spec, (int, float, np.floating, np.integer)):
            return float(spec)
        if isinstance(spec, (list, tuple, np.ndarray)) and len(spec) == 2:
            lo, hi = float(spec[0]), float(spec[1])
            return float(self._rng.uniform(lo, hi))
        raise TypeError(f"cannot interpret scalar spec: {spec!r}")

    def _draw_direction(self, spec):
        """Sample a direction once per episode from a name, vector, or list."""
        if isinstance(spec, list) and spec and not isinstance(spec[0], (int, float)):
            choice = spec[self._rng.integers(len(spec))]
            return resolve_direction(choice, self._directions)
        if isinstance(spec, list) and len(spec) == 3:
            return resolve_direction(spec, self._directions)
        return resolve_direction(spec, self._directions)

    def _draw_seed(self) -> int:
        return int(self._rng.integers(0, 2**31 - 1))