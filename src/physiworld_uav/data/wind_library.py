# ═══════════════════════════════════════════════════════════════════════════
# WindLibrary
# ═══════════════════════════════════════════════════════════════════════════


import math
import numpy as np
from typing import Any, Dict
from .utils import _sample, _resolve_direction, _to_list


class WindLibrary:
    """
    Factory for all wind types described in the config.

    Each wind spec is resolved at every step to a world-frame force vector
    (Newtons). The magnitude is scaled by `ood.wind_scale`; when
    `ood.wind_directions_override` is set, it replaces the per-segment
    direction.
    """

    def __init__(self, cfg, device: str = "cpu", seed: int = 42):
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(seed + 7919)  # distinct stream

        # OOD modifiers
        self.ood = {
            "wind_scale": float(getattr(cfg.ood, "wind_scale", 1.0)),
            "wind_directions_override": getattr(cfg.ood, "wind_directions_override", None),
            "temporal_speed_scale": float(getattr(cfg.ood, "temporal_speed_scale", 1.0)),
        }

        # Direction tables from every mode's `directions` block
        self.directions = {}
        for mode in ("translation", "yaw", "attitude", "combined", "hover"):
            try:
                d = cfg.dataset.modes[mode].definition.directions
                self.directions.update(dict(d))
            except Exception:
                pass
        if not self.directions:
            self.directions = {
                "north": [1, 0, 0], "south": [-1, 0, 0],
                "east": [0, 1, 0], "west": [0, -1, 0],
                "up": [0, 0, 1], "down": [0, 0, -1],
            }

        # Per-trajectory state for stateful winds (dryden, gust_mid timing, random_multi)
        self._wind_state: Dict[Any, Dict[str, Any]] = {}

    def clear_wind_state(self):
        """Clear all per-trajectory wind state (for new episode)."""
        self._wind_state.clear()
        
    # ────────────────────────────────────────────────────────────────
    def compute(
        self,
        spec: dict,
        t: float,
        dt: float,
        phase_elapsed: float,
        phase_duration: float,
        state_key: Any,
    ) -> np.ndarray:
        """
        Compute the world-frame force vector (N) at time t.

        Parameters
        ----------
        spec : dict
            Wind spec from the config.
        t : float
            Absolute time from trajectory start (for `step_time_s`, etc.).
        dt : float
            Simulation timestep.
        phase_elapsed : float
            Time within the current phase (for `gust_mid.trigger_at_progress`).
        phase_duration : float
            Total duration of the phase (for `gust_mid.trigger_at_progress`).
        state_key : hashable
            Key for storing stateful wind progress (dryden's prev force).

        Returns
        -------
        force : np.ndarray, shape (3,), world frame, Newtons.
        """
        wtype = spec.get("type", "none")
        method = {
            "none": self._none,
            "constant": self._constant,
            "step": self._step,
            "multi_step": self._multi_step,
            "sinusoidal": self._sinusoidal,
            "gust": self._gust,
            "gust_mid": self._gust_mid,
            "dryden": self._dryden,
            "rotating": self._rotating,
            "random": self._random,
            "random_multi": self._random_multi,
        }.get(wtype)
        if method is None:
            raise ValueError(f"Unknown wind type: {wtype!r}")

        if state_key not in self._wind_state:
            self._wind_state[state_key] = {}
        wstate = self._wind_state[state_key]

        force = method(spec, t, dt, phase_elapsed, phase_duration, wstate)
        return force.astype(np.float32)

    # ────────────────────────────────────────────────────────────────
    def _apply_ood(self, force: np.ndarray, direction_spec) -> np.ndarray:
        """Apply wind_scale and direction override."""
        scaled = force * self.ood["wind_scale"]
        if self.ood["wind_directions_override"] is not None:
            override_dir = _resolve_direction(
                self.ood["wind_directions_override"], self.directions, self.rng
            )
            mag = float(np.linalg.norm(scaled))
            scaled = override_dir * mag
        return scaled

    def _resolve_direction(self, spec, t=0.0, wstate=None) -> np.ndarray:
        """Resolve direction spec to a unit vector, handling rotating wind state."""
        if isinstance(spec, dict) or (hasattr(spec, "get") and callable(getattr(spec, "get"))):
            # Could be a rotating-direction spec
            if "name" in spec and spec["name"] == "rotating":
                rate = float(spec.get("rotation_rate_radps", 0.3))
                theta0 = float(spec.get("initial_direction_rad", 0.0))
                theta = theta0 + rate * t
                return np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float32)
        return _resolve_direction(spec, self.directions, self.rng)

    # ────────────────────────────────────────────────────────────────
    # Individual wind types
    # ────────────────────────────────────────────────────────────────
    def _none(self, spec, t, dt, phase_elapsed, phase_duration, wstate):
        return np.zeros(3, dtype=np.float32)

    def _constant(self, spec, t, dt, phase_elapsed, phase_duration, wstate):
        mag = _sample(self.rng, spec.get("magnitude_N", 2.0))
        dir_spec = spec.get("direction", spec.get("direction_from_list", [1, 0, 0]))
        d = self._resolve_direction(dir_spec, t)
        return self._apply_ood(d * mag, dir_spec)

    def _step(self, spec, t, dt, phase_elapsed, phase_duration, wstate):
        step_time = float(spec.get("step_time_s", 0.0))
        if t < step_time:
            mag = float(spec.get("from_magnitude_N", 0.0))
        else:
            mag = float(spec.get("to_magnitude_N", 4.0))
        d = self._resolve_direction(spec.get("direction", [1, 0, 0]), t)
        return self._apply_ood(d * mag, spec.get("direction"))

    def _multi_step(self, spec, t, dt, phase_elapsed, phase_duration, wstate):
        steps = spec.get("steps", [])
        mag = 0.0
        for s in steps:
            if t >= float(s.get("t_s", 0.0)):
                mag = float(s.get("magnitude_N", 0.0))
        d = self._resolve_direction(
            spec.get("direction", spec.get("direction_from_list", [1, 0, 0])), t
        )
        return self._apply_ood(d * mag, spec.get("direction"))

    def _sinusoidal(self, spec, t, dt, phase_elapsed, phase_duration, wstate):
        amp = _sample(self.rng, spec.get("amplitude_N", 3.0))
        freq = _sample(self.rng, spec.get("frequency_hz", 0.2)) \
            * self.ood["temporal_speed_scale"]
        scalar = amp * math.sin(2 * math.pi * freq * t)
        d = self._resolve_direction(spec.get("direction", [1, 0, 0]), t)
        return self._apply_ood(d * scalar, spec.get("direction"))

    def _gust(self, spec, t, dt, phase_elapsed, phase_duration, wstate):
        gust_t = float(spec.get("gust_time_s", 0.0))
        dur = float(spec.get("duration_s", 0.5))
        if gust_t <= t < gust_t + dur:
            mag = float(spec.get("peak_magnitude_N", 6.0))
        else:
            mag = float(spec.get("from_magnitude_N", 2.0))
        d = self._resolve_direction(spec.get("direction", [1, 0, 0]), t)
        return self._apply_ood(d * mag, spec.get("direction"))

    def _gust_mid(self, spec, t, dt, phase_elapsed, phase_duration, wstate):
        if "gust_trigger" not in wstate:
            prog = _sample(self.rng, spec.get("trigger_at_progress", [0.3, 0.7]))
            wstate["gust_trigger"] = prog * phase_duration
            wstate["gust_dur"] = _sample(self.rng, spec.get("duration_s", [0.3, 0.8]))
            wstate["gust_peak"] = _sample(self.rng, spec.get("peak_magnitude_N", [5.0, 8.0]))
        gt = wstate["gust_trigger"]
        dur = wstate["gust_dur"]
        if gt <= phase_elapsed < gt + dur:
            mag = wstate["gust_peak"]
        else:
            mag = 0.0
        d = self._resolve_direction(
            spec.get("direction", spec.get("direction_from_list", [1, 0, 0])), t
        )
        return self._apply_ood(d * mag, spec.get("direction"))

    def _dryden(self, spec, t, dt, phase_elapsed, phase_duration, wstate):
        sigma = _sample(self.rng, spec.get("sigma", 0.8))
        tau = max(_sample(self.rng, spec.get("length_scale_m", 100.0)) / 10.0, float(dt), 1e-4)
        alpha = math.exp(-dt / tau)
        prev = wstate.get("dryden_force", np.zeros(3, dtype=np.float32))
        noise = self.rng.normal(0.0, sigma, size=3).astype(np.float32)
        new_force = alpha * prev + math.sqrt(max(0.0, 1.0 - alpha * alpha)) * noise
        wstate["dryden_force"] = new_force
        return self._apply_ood(new_force, None)

    def _rotating(self, spec, t, dt, phase_elapsed, phase_duration, wstate):
        mag = _sample(self.rng, spec.get("magnitude_N", 2.0))
        rate = _sample(self.rng, spec.get("rotation_rate_radps", 0.4))
        theta0 = _sample(self.rng, spec.get("initial_direction_rad", 0.0))
        theta = theta0 + rate * t
        d = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float32)
        return self._apply_ood(d * mag, None)

    def _random(self, spec, t, dt, phase_elapsed, phase_duration, wstate):
        sigma = _sample(self.rng, spec.get("sigma", 1.0))
        return self.rng.normal(0.0, sigma, size=3).astype(np.float32)

    def _random_multi(self, spec, t, dt, phase_elapsed, phase_duration, wstate):
        # Sample type/magnitude/direction on resample boundaries
        if "rm_next_t_change" not in wstate:
            wstate["rm_next_t_change"] = 0.0
            wstate["rm_next_dir_change"] = 0.0
            wstate["rm_current_type"] = "constant"
            wstate["rm_current_mag"] = 0.0
            wstate["rm_current_dir"] = np.array([1, 0, 0], dtype=np.float32)

        allowed = _to_list(spec.get("allowed_types", ["constant", "step", "sinusoidal", "gust", "dryden"]))
        mag_range = _to_list(spec.get("magnitude_range_N", [0.0, 6.0]))
        type_interval = _to_list(spec.get("type_change_interval_s", [5.0, 10.0]))
        dir_interval = _to_list(spec.get("direction_change_interval_s", [3.0, 6.0]))

        if t >= wstate["rm_next_t_change"]:
            wstate["rm_current_type"] = allowed[self.rng.integers(len(allowed))]
            wstate["rm_next_t_change"] = t + self.rng.uniform(type_interval[0], type_interval[1])
            wstate["rm_current_mag"] = self.rng.uniform(mag_range[0], mag_range[1])
            wstate["rm_next_dir_change"] = 0.0  # force direction resample

        if t >= wstate["rm_next_dir_change"]:
            # Pick any direction name from the table
            names = list(self.directions.keys())
            chosen_name = names[self.rng.integers(len(names))]
            wstate["rm_current_dir"] = np.array(self.directions[chosen_name], dtype=np.float32)
            wstate["rm_next_dir_change"] = t + self.rng.uniform(dir_interval[0], dir_interval[1])

        # Apply magnitude based on type
        wtype = wstate["rm_current_type"]
        base_mag = wstate["rm_current_mag"]
        if wtype == "constant":
            mag = base_mag
        elif wtype == "sinusoidal":
            mag = base_mag * math.sin(2 * math.pi * 0.2 * t)
        elif wtype == "gust":
            mag = base_mag if (int(t * 2) % 10 < 2) else 0.0
        elif wtype == "dryden":
            prev = wstate.get("rm_dryden_force", np.zeros(3, dtype=np.float32))
            tau = 10.0
            alpha = math.exp(-dt / tau)
            noise = self.rng.normal(0.0, base_mag * 0.3, size=3).astype(np.float32)
            mag_force = alpha * prev + math.sqrt(max(0.0, 1 - alpha * alpha)) * noise
            wstate["rm_dryden_force"] = mag_force
            return mag_force
        else:  # step
            mag = base_mag if t > 5.0 else 0.0

        force = wstate["rm_current_dir"] * float(mag)
        return self._apply_ood(force, None)