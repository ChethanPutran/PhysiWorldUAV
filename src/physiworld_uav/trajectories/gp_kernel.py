"""
periodic + composite kernel
""""""Periodic and composite GP kernels with analytic derivatives (Task §6).

The periodic kernel is

    k(tau) = sigma^2 * exp( -2 sin^2(pi |tau| / p) / ell^2 )

We expose ``k``, ``k_t``, ``k_tt`` (first and second derivatives in
*absolute* time-difference), and derive the four cross-derivative blocks
needed for joint (position, velocity, acceleration) sampling by
relation to ``g = k(|tau|)``:

    k_ff(t_i, t_j) = g(t_i - t_j)
    k_fv           = -g'(t_i - t_j)      (derivative w.r.t. t_j)
    k_fa           =  g''(t_i - t_j)
    k_vv           = -g''(t_i - t_j)
    k_va           =  g'''(t_i - t_j)    (finite-differenced)
    k_aa           =  g''''(t_i - t_j)   (finite-differenced)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PeriodicKernel:
    sigma: float
    ell: float
    period: float

    def _coeffs(self) -> tuple[float, float, float, float]:
        # A = sigma^2 exp(-1/ell^2); B = 1/ell^2; w = 2*pi/p
        A = self.sigma ** 2 * np.exp(-1.0 / self.ell ** 2)
        B = 1.0 / self.ell ** 2
        w = 2.0 * np.pi / self.period
        return A, B, w

    # ---- base kernel and derivatives w.r.t. absolute time-difference ------
    def k(self, tau: np.ndarray) -> np.ndarray:
        A, B, w = self._coeffs()
        return A * np.exp(B * np.cos(w * tau))

    def k_t(self, tau: np.ndarray) -> np.ndarray:
        A, B, w = self._coeffs()
        return -A * B * w * np.sin(w * tau) * np.exp(B * np.cos(w * tau))

    def k_tt(self, tau: np.ndarray) -> np.ndarray:
        A, B, w = self._coeffs()
        u = B * np.cos(w * tau)
        return -A * B * w ** 2 * (np.cos(w * tau) - B * np.sin(w * tau) ** 2) * np.exp(u)

    def k_ttt(self, tau: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        return (self.k_tt(tau + eps) - self.k_tt(tau - eps)) / (2.0 * eps)

    def k_tttt(self, tau: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        return (self.k_tt(tau + eps) - 2.0 * self.k_tt(tau) + self.k_tt(tau - eps)) / (eps ** 2)


@dataclass(frozen=True)
class CompositeKernel:
    """Sum of PeriodicKernels with different (ell, p) and total prior variance ``sigma²``.

    Each of the n components gets variance ``sigma²/n`` (plan §6.3-T7: the
    previous version gave every component ``sigma²``, i.e. total ``n·sigma²``).
    Note: for the periodic kernel k(0) = sigma²; the marginal variance is exact.
    """

    sigma: float
    components: tuple[tuple[float, float], ...] = field(default=())  # (ell, period)

    def _kernels(self) -> list[PeriodicKernel]:
        n = max(1, len(self.components))
        return [PeriodicKernel(self.sigma / np.sqrt(n), ell, p) for ell, p in self.components]

    def k(self, tau):      return sum(k.k(tau) for k in self._kernels())
    def k_t(self, tau):    return sum(k.k_t(tau) for k in self._kernels())
    def k_tt(self, tau):   return sum(k.k_tt(tau) for k in self._kernels())
    def k_ttt(self, tau):  return sum(k.k_ttt(tau) for k in self._kernels())
    def k_tttt(self, tau): return sum(k.k_tttt(tau) for k in self._kernels())


def make_composite(
    sigma: float,
    length_scales: list[float],
    periods: list[float],
) -> CompositeKernel:
    if len(length_scales) != len(periods):
        raise ValueError("length_scales and periods must have equal length")
    return CompositeKernel(
        sigma=sigma,
        components=tuple(zip(length_scales, periods)),
    )