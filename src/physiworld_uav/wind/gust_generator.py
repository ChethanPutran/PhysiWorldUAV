"""Gust models — design doc §7: 1-cosine, step, sinusoidal, random events, OU."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import resolve_direction


# ── 1-cosine ────────────────────────────────────────────────────────────────
class CosineGust:
    """1-cosine gust over the window ``[t_start, t_start + 2 * T_rise]``.

    Equation (FAR Part 25 / MIL-F-8785C):

        u_g(t) = 0.5 * U_peak * (1 - cos(pi * (t - t_start) / T_rise))

    The gust rises from 0 to ``U_peak`` over ``T_rise`` seconds and falls
    back to 0 over the next ``T_rise`` seconds.
    """

    name = "gust_cosine"
    is_stateful = False

    def __init__(
        self,
        peak_mps: float,
        start_time_s: float,
        rise_time_s: float,
        direction,
        directions_table=None,
    ) -> None:
        if rise_time_s <= 0:
            raise ValueError("rise_time_s must be > 0")
        if start_time_s < 0:
            raise ValueError("start_time_s must be >= 0")
        self.peak = float(peak_mps)
        self.t_start = float(start_time_s)
        self.T_rise = float(rise_time_s)
        self.dir = resolve_direction(direction, directions_table)

    def contribution(self, position_W, time_s, dt=None) -> np.ndarray:
        tau = time_s - self.t_start
        if tau < 0.0 or tau > 2.0 * self.T_rise:
            return np.zeros(3)
        scalar = 0.5 * self.peak * (1.0 - np.cos(np.pi * tau / self.T_rise))
        return scalar * self.dir

    def reset(self, seed=None) -> None:
        pass


# ── step ────────────────────────────────────────────────────────────────────
class StepGust:
    """Step change in wind magnitude at ``t_step``.

    ``rise_time_s = 0`` is an ideal step; ``rise_time_s > 0`` is a linear ramp
    from ``before`` to ``after`` over ``[t_step, t_step + rise_time_s]``
    (controllable sharpness for the temporal-OOD class, plan §6.2-W1).
    """

    name = "gust_step"
    is_stateful = False

    def __init__(self, before_mps: float, after_mps: float, step_time_s: float,
                 direction, directions_table=None, rise_time_s: float = 0.0) -> None:
        if rise_time_s < 0:
            raise ValueError("rise_time_s must be >= 0")
        self.before = float(before_mps)
        self.after = float(after_mps)
        self.t_step = float(step_time_s)
        self.rise = float(rise_time_s)
        self.dir = resolve_direction(direction, directions_table)

    def contribution(self, position_W, time_s, dt=None) -> np.ndarray:
        if time_s < self.t_step:
            mag = self.before
        elif self.rise <= 0.0 or time_s >= self.t_step + self.rise:
            mag = self.after
        else:
            frac = (time_s - self.t_step) / self.rise
            mag = self.before + (self.after - self.before) * frac
        return mag * self.dir

    def reset(self, seed=None) -> None:
        pass


# ── sinusoidal ──────────────────────────────────────────────────────────────
class SinusoidalGust:
    """Oscillating gust around ``mean_mps`` along a fixed direction."""

    name = "gust_sinusoidal"
    is_stateful = False

    def __init__(self, amplitude_mps: float, frequency_hz: float,
                 mean_mps: float, direction, directions_table=None) -> None:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be > 0")
        self.amp = float(amplitude_mps)
        self.f = float(frequency_hz)
        self.mean = float(mean_mps)
        self.dir = resolve_direction(direction, directions_table)

    def contribution(self, position_W, time_s, dt=None) -> np.ndarray:
        mag = self.mean + self.amp * np.sin(2.0 * np.pi * self.f * time_s)
        return mag * self.dir

    def reset(self, seed=None) -> None:
        pass


# ── Ornstein–Uhlenbeck ──────────────────────────────────────────────────────
class OUGust:
    """Ornstein–Uhlenbeck stochastic gust (3 axes), stationary from t = 0 (plan §6.2-W2).

    Continuous: ``dX = -X/tau dt + sqrt(2/tau) dW`` per standardised axis; output
    ``mean + sigma ⊙ X`` with ``sigma = (σ_h, σ_h, vertical_ratio·σ_h)`` (world frame).

    Exact discrete update (independent of dt):
        X[k+1] = a X[k] + sqrt(1 − a²) N(0, I),   a = exp(−dt/tau)
    The state is drawn from the stationary distribution N(0, I) at reset.
    """

    name = "gust_ou"
    is_stateful = True

    def __init__(self, sigma_mps: float, tau_s: float, mean_mps=(0.0, 0.0, 0.0),
                 seed: int | None = None, vertical_ratio: float = 0.5) -> None:
        if sigma_mps < 0:
            raise ValueError("sigma_mps must be >= 0")
        if tau_s <= 0:
            raise ValueError("tau_s must be > 0")
        if vertical_ratio < 0:
            raise ValueError("vertical_ratio must be >= 0")
        self.sigma = float(sigma_mps) * np.array([1.0, 1.0, float(vertical_ratio)])
        self.tau = float(tau_s)
        self.mean = np.asarray(mean_mps, dtype=np.float64).reshape(3)
        self._rng = np.random.default_rng(seed)
        self._x = self._rng.standard_normal(3)
        self._last_t: float | None = None

    def contribution(self, position_W, time_s, dt=None) -> np.ndarray:
        if self._last_t is None:
            self._last_t = time_s
            return self.mean + self.sigma * self._x
        if dt is None:
            dt = time_s - self._last_t
        if dt <= 0.0:
            return self.mean + self.sigma * self._x
        a = np.exp(-dt / self.tau)
        self._x = a * self._x + np.sqrt(max(0.0, 1.0 - a * a)) * self._rng.standard_normal(3)
        self._last_t = time_s
        return self.mean + self.sigma * self._x

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._x = self._rng.standard_normal(3)
        self._last_t = None


# ── random gust events ──────────────────────────────────────────────────────
class RandomGustSequence:
    """Pre-sampled sequence of 1-cosine pulses (design doc: random gust events).

    At ``reset(seed)`` we draw ``n_events`` gust start times from a Poisson
    process, then amplitudes and directions uniformly from the configured
    ranges. Evaluation is a sum of the individual 1-cosine pulses.
    """

    name = "gust_events"
    is_stateful = True

    def __init__(self, rate_hz: float, peak_range_mps: tuple[float, float],
                 rise_range_s: tuple[float, float],
                 duration_s: float, directions_table=None,
                 seed: int | None = None) -> None:
        self.rate = float(rate_hz)
        self.peak_range = peak_range_mps
        self.rise_range = rise_range_s
        self.duration = float(duration_s)
        self._directions_table = directions_table
        self._rng = np.random.default_rng(seed)
        self._events: list[tuple[float, float, float, np.ndarray]] = []
        self._sampled = False

    def _sample_events(self) -> None:
        n = self._rng.poisson(self.rate * self.duration)
        events = []
        for _ in range(n):
            t0 = float(self._rng.uniform(0.0, self.duration))
            peak = float(self._rng.uniform(*self.peak_range))
            rise = float(self._rng.uniform(*self.rise_range))
            # isotropic XY direction, no vertical
            theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
            d = np.array([np.cos(theta), np.sin(theta), 0.0])
            events.append((t0, peak, rise, d))
        self._events = events
        self._sampled = True

    def contribution(self, position_W, time_s, dt=None) -> np.ndarray:
        if not self._sampled:
            self._sample_events()
        total = np.zeros(3)
        for t0, peak, rise, d in self._events:
            tau = time_s - t0
            if 0.0 <= tau <= 2.0 * rise:
                total += 0.5 * peak * (1.0 - np.cos(np.pi * tau / rise)) * d
        return total

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._events = []
        self._sampled = False