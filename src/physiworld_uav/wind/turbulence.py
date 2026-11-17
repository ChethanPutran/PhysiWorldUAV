"""Dryden turbulence, MIL-F-8785C (plan §6.2-W3). Rewritten in Run 4.

Transfer functions (T_c = L_c / V, unit-variance white noise input):

    H_u(s) ∝ 1 / (1 + T_u s)
    H_v(s) ∝ (1 + √3 T_v s) / (1 + T_v s)²        (same form for w)

Implementation:

* each axis is a state-space system ẋ = A x + B η, y = C x;
* **exact** zero-order-hold discretisation of the stochastic system with Van
  Loan's method (A_d = e^{A dt}, Q_d = ∫ e^{As} B Bᵀ e^{Aᵀs} ds);
* the stationary covariance P solves the continuous Lyapunov equation
  A P + P Aᵀ + B Bᵀ = 0 (closed form, no iteration) and the output is scaled
  so that its stationary standard deviation equals σ_c **exactly**;
* the state is initialised from the stationary distribution N(0, P), so the
  process is stationary from t = 0;
* components are generated in the mean-wind frame (u along the horizontal mean
  wind, v lateral, w vertical) and rotated to the world frame.

Frozen-turbulence speed: V = max(‖mean horizontal wind‖, V_min), fixed per
episode (Taylor's hypothesis with advection by the mean wind; a near-hovering
vehicle does not provide the airspeed the aircraft form assumes). Making V
depend on the vehicle's own velocity would make the realisation depend on the
trajectory and break counterfactual pairing — documented deviation (Run 4 report).

Low-altitude parameters (``low_altitude``): MIL-F-8785C with h in feet,
L_w = h, L_u = L_v = h / (0.177 + 0.000823 h)^1.2, σ_w = 0.1 W₂₀,
σ_u/σ_w = σ_v/σ_w = 1 / (0.177 + 0.000823 h)^0.4, valid for h ≥ 10 ft
(clamped there; using it at 1–3 m altitude is an extrapolation).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

FT = 0.3048


def low_altitude(h_m: float, w20_mps: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(σ_u, σ_v, σ_w) [m/s] and (L_u, L_v, L_w) [m] per MIL-F-8785C low altitude."""
    h = max(h_m / FT, 10.0)
    k = 0.177 + 0.000823 * h
    L_uv = h / k ** 1.2 * FT
    sigma_w = 0.1 * w20_mps
    sigma_uv = sigma_w / k ** 0.4
    return (sigma_uv, sigma_uv, sigma_w), (L_uv, L_uv, h * FT)


def _first_order(T: float):
    A = np.array([[-1.0 / T]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    return A, B, C


def _second_order(T: float):
    A = np.array([[0.0, 1.0], [-1.0 / T ** 2, -2.0 / T]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0 / T ** 2, math.sqrt(3.0) / T]])
    return A, B, C


def _van_loan(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = B @ B.T
    M[n:, n:] = A.T
    E = expm(M * dt)
    Ad = E[n:, n:].T
    Qd = Ad @ E[:n, n:]
    return Ad, 0.5 * (Qd + Qd.T)


class _Axis:
    def __init__(self, order: int, T: float, sigma: float) -> None:
        self.A, self.B, self.C = (_first_order if order == 1 else _second_order)(T)
        P = solve_continuous_lyapunov(self.A, -self.B @ self.B.T)
        self.P = 0.5 * (P + P.T)
        var_y = float((self.C @ self.P @ self.C.T)[0, 0])
        self.scale = sigma / math.sqrt(var_y) if sigma > 0 else 0.0
        self._dt = None
        self.x = np.zeros(self.A.shape[0])

    def init(self, rng: np.random.Generator) -> None:
        L = np.linalg.cholesky(self.P + 1e-30 * np.eye(self.P.shape[0]))
        self.x = L @ rng.standard_normal(self.P.shape[0])

    def step(self, dt: float, rng: np.random.Generator) -> None:
        if self._dt != dt:
            self.Ad, Qd = _van_loan(self.A, self.B, dt)
            w, V = np.linalg.eigh(Qd)
            self.Lq = V * np.sqrt(np.clip(w, 0.0, None))
            self._dt = dt
        self.x = self.Ad @ self.x + self.Lq @ rng.standard_normal(self.x.shape[0])

    def output(self) -> float:
        return self.scale * float((self.C @ self.x)[0])


class DrydenTurbulence:
    """MIL-F-8785C Dryden turbulence (stateful, stationary from t = 0)."""

    name = "dryden"
    is_stateful = True

    def __init__(self, airspeed_mps: float, intensities_mps: tuple[float, float, float],
                 length_scales_m: tuple[float, float, float], mean_W: np.ndarray | None = None,
                 seed: int | None = None, mean_direction_W: np.ndarray | None = None,
                 min_airspeed_mps: float = 1.0) -> None:
        if airspeed_mps < 0:
            raise ValueError("airspeed_mps must be >= 0")
        if any(s < 0 for s in intensities_mps):
            raise ValueError("intensities must be >= 0")
        if any(L <= 0 for L in length_scales_m):
            raise ValueError("length_scales must be > 0")
        self.V = max(float(airspeed_mps), float(min_airspeed_mps))
        self.sigma = np.asarray(intensities_mps, dtype=np.float64)
        self.L = np.asarray(length_scales_m, dtype=np.float64)
        self.T = self.L / self.V
        self.mean = np.zeros(3) if mean_W is None else np.asarray(mean_W, dtype=np.float64).reshape(3)
        d = np.array([1.0, 0.0, 0.0]) if mean_direction_W is None else np.asarray(mean_direction_W, float)
        d = np.array([d[0], d[1], 0.0])
        d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else np.array([1.0, 0.0, 0.0])
        lat = np.array([-d[1], d[0], 0.0])
        self._R = np.stack([d, lat, np.array([0.0, 0.0, 1.0])], axis=1)   # mean-wind frame → world
        self._axes = [_Axis(1, self.T[0], self.sigma[0]), _Axis(2, self.T[1], self.sigma[1]),
                      _Axis(2, self.T[2], self.sigma[2])]
        self._rng = np.random.default_rng(seed)
        self._last_t: float | None = None
        self._init()

    def _init(self) -> None:
        for ax in self._axes:
            ax.init(self._rng)

    def contribution(self, position_W, time_s, dt=None) -> np.ndarray:
        if self._last_t is not None:
            step = (time_s - self._last_t) if dt is None else dt
            if step > 0.0:
                for ax in self._axes:
                    ax.step(step, self._rng)
        self._last_t = time_s
        return self.mean + self._R @ np.array([ax.output() for ax in self._axes])

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._last_t = None
        self._init()
