"""Cholesky-based joint sampling of (position, velocity, acceleration) (Task §6)."""
from __future__ import annotations

import numpy as np

from .gp_kernel import CompositeKernel


def _diff_matrix(t: np.ndarray) -> np.ndarray:
    return t[:, None] - t[None, :]


def joint_covariance(kernel: CompositeKernel, t: np.ndarray, jitter: float) -> np.ndarray:
    """Build the 3N x 3N covariance over [p, v, a] at times ``t``.

    Uses the derivative relations documented in ``gp_kernel.py``.
    """
    N = t.size
    tau = _diff_matrix(t)
    g   = kernel.k(tau)
    gp  = kernel.k_t(tau)
    gpp = kernel.k_tt(tau)
    gppp  = kernel.k_ttt(tau)
    gpppp = kernel.k_tttt(tau)

    kff = g
    kfv = -gp
    kfa = gpp
    kvv = -gpp
    kva = gppp
    kaa = gpppp

    C = np.zeros((3 * N, 3 * N))
    C[0:N, 0:N]   = kff
    C[0:N, N:2 * N] = kfv
    C[0:N, 2 * N:3 * N] = kfa
    C[N:2 * N, 0:N] = kfv.T
    C[N:2 * N, N:2 * N] = kvv
    C[N:2 * N, 2 * N:3 * N] = kva
    C[2 * N:3 * N, 0:N] = kfa.T
    C[2 * N:3 * N, N:2 * N] = kva.T
    C[2 * N:3 * N, 2 * N:3 * N] = kaa

    # Symmetrise to kill floating-point asymmetry before Cholesky.
    C = 0.5 * (C + C.T)
    C.flat[:: C.shape[0] + 1] += jitter
    return C


def _factor(C: np.ndarray, rel_jitter: float, max_rel_jitter: float = 1e-3) -> tuple[np.ndarray, str]:
    """Return L with L Lᵀ ≈ C and the method used.

    Jitter is *relative* to the mean diagonal (the joint p/v/a covariance spans
    many orders of magnitude and is numerically low-rank, plan §6.3-T7); it is
    escalated ×10 on failure, then falls back to an eigendecomposition with
    negative eigenvalues clipped to zero.
    """
    scale = float(np.mean(np.diag(C)))
    eps = rel_jitter
    while eps <= max_rel_jitter:
        try:
            Cj = C.copy()
            Cj.flat[:: C.shape[0] + 1] += eps * scale
            return np.linalg.cholesky(Cj), f"cholesky(rel_jitter={eps:g})"
        except np.linalg.LinAlgError:
            eps *= 10.0
    w, V = np.linalg.eigh(C)
    return V * np.sqrt(np.clip(w, 0.0, None)), "eigh(clipped)"


def sample_joint_gp(
    kernel: CompositeKernel,
    t: np.ndarray,
    mean: np.ndarray | None = None,
    jitter: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(position, velocity, acceleration)`` of shape (N, 3).

    ``jitter`` is relative to the mean prior variance. The three axes are
    independent draws from the same (once-factorised) covariance.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = t.size
    C = joint_covariance(kernel, t, 0.0)
    L, _ = _factor(C, jitter)
    Z = rng.standard_normal((L.shape[1], 3))
    S = L @ Z                                            # (3N, 3)
    pos, vel, acc = S[0:N], S[N:2 * N], S[2 * N:3 * N]
    if mean is not None:
        pos = pos + np.asarray(mean, dtype=float)[None, :]
    return pos, vel, acc
