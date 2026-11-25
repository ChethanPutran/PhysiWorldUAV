"""Paired model comparisons (e.g. the B3–B6 gap), residual diagnostics and the residual-off test.

* ``paired_gap``: RMSE of two models on the SAME windows, their difference and ratio per horizon step, with a bootstrap CI
  that resamples whole episodes (windows of one episode overlap and are not independent).
* ``residual_diagnostics``: size of d_θ against f_phys along B3's own rollout, in physical units.
* ``residual_off``: context manager that zeroes d_θ (the learned latent coupling stays): "B3 without its residual".
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluator import BATCH_KEYS, _to


def rmse_curve(pe: np.ndarray, ks: Sequence[int]) -> dict[int, float]:
    return {k: float(np.sqrt((pe[:, k - 1] ** 2).mean())) for k in ks}


def _per_episode(pe: np.ndarray, eids: np.ndarray, ks: Sequence[int]):
    ids = sorted(set(eids.tolist()))
    ss = np.array([[(pe[eids == e, k - 1] ** 2).sum() for k in ks] for e in ids])       # (E, K) sum of squares
    n = np.array([(eids == e).sum() for e in ids], dtype=float)                           # (E,) windows
    return ids, ss, n


def paired_gap(err_a: np.ndarray, err_b: np.ndarray, eids: np.ndarray, ks: Sequence[int], n_boot: int = 1000, seed: int = 0) -> dict:
    """``err_*``: (N, K) position errors of model A and B on the same N windows. Gap = RMSE_A − RMSE_B (positive: A worse)."""
    if err_a.shape != err_b.shape or len(eids) != len(err_a):
        raise ValueError("the two models must be evaluated on identical windows")
    _, ssa, n = _per_episode(err_a, eids, ks)
    _, ssb, _ = _per_episode(err_b, eids, ks)
    ra, rb = np.sqrt(ssa.sum(0) / n.sum()), np.sqrt(ssb.sum(0) / n.sum())
    out = {"ks": list(ks), "rmse_a": ra.tolist(), "rmse_b": rb.tolist(), "gap": (ra - rb).tolist(),
           "ratio": (ra / np.maximum(rb, 1e-12)).tolist(), "n_windows": int(n.sum()), "n_episodes": int(len(n))}
    if len(n) > 1 and n_boot:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(n), (n_boot, len(n)))
        den = n[idx].sum(1)[:, None]
        ga = np.sqrt(ssa[idx].sum(1) / den) - np.sqrt(ssb[idx].sum(1) / den)
        out["gap_ci95"] = [np.percentile(ga, 2.5, axis=0).tolist(), np.percentile(ga, 97.5, axis=0).tolist()]
    return out


# ----------------------------------------------------------------------------------- residual
@contextmanager
def residual_off(model):
    """Temporarily zero d_θ of a B3-type model (its mask buffer), restoring it afterwards."""
    if not hasattr(model, "res_mask"):
        raise ValueError("model has no residual")
    saved = model.res_mask.clone()
    model.res_mask.zero_()
    try:
        yield model
    finally:
        model.res_mask.copy_(saved)


@torch.no_grad()
def residual_diagnostics(model, dataset, *, horizon: int, device="cpu", batch_size: int = 512) -> dict:
    """Per-window, per-step magnitudes along B3's open-loop rollout (physical units):

    ``r_acc`` ‖d_θ,v̇‖ [m/s²], ``r_ang`` ‖d_θ,ω̇‖ [rad/s²], ``f_net`` ‖v̇ of f_phys‖ (thrust − gravity: ≈ 0 near hover),
    ``f_spec`` ‖specific force of f_phys‖ = ‖v̇ + g ẑ‖ (≈ g near hover), ``f_ang`` ‖ω̇ of f_phys‖. Ratios to read:
    ``r_acc/f_spec`` (size of the learned correction against the thrust-scale acceleration) and ``r_ang/f_ang``."""
    if not (hasattr(model, "residual") and hasattr(model, "physics") and hasattr(model, "observe")):
        raise ValueError("residual diagnostics need a latent physics+residual model (b3)")
    model.eval().to(device)
    g = model.physics.g
    acc = {k: [] for k in ("r_acc", "r_ang", "f_net", "f_spec", "f_ang")}
    eids = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        b = _to(batch, device)
        out = model(b, horizon)
        s_before = torch.cat([b["state_history"][:, -1:], out["pred"][:, :-1]], 1)             # (B,K,13), absolute positions
        u = b["future_actions"][:, :horizon]
        B, K = u.shape[:2]
        h = out["h_traj"].reshape(B * K, -1)
        s, uu = s_before.reshape(B * K, 13), u.reshape(B * K, 4)
        r = model.residual(h, s, uu)
        f = model.physics.derivative(s, uu)
        zhat = torch.tensor([0.0, 0.0, g], device=s.device)
        for key, val in (("r_acc", r[:, 3:6]), ("r_ang", r[:, 9:12]), ("f_net", f[:, 3:6]), ("f_spec", f[:, 3:6] + zhat),
                         ("f_ang", f[:, 10:13])):
            acc[key].append(val.norm(dim=-1).reshape(B, K).cpu())
        eids += list(batch["episode_id"])
    return {**{k: torch.cat(v).numpy() for k, v in acc.items()}, "eids": np.array(eids)}


def summarize_residual(diag: dict, mask: np.ndarray | None = None) -> dict:
    sel = slice(None) if mask is None else mask
    m = {k: v[sel].mean(0) for k, v in diag.items() if k != "eids"}
    return {"mean_r_acc_mps2": m["r_acc"].tolist(), "mean_r_ang_radps2": m["r_ang"].tolist(),
            "mean_f_spec_mps2": float(m["f_spec"].mean()), "mean_f_net_mps2": float(m["f_net"].mean()),
            "mean_f_ang_radps2": float(m["f_ang"].mean()),
            "ratio_r_acc_to_f_spec": (m["r_acc"] / np.maximum(m["f_spec"], 1e-9)).tolist(),
            "ratio_r_ang_to_f_ang": (m["r_ang"] / np.maximum(m["f_ang"], 1e-9)).tolist(),
            "n_windows": int(len(diag["r_acc"][sel]))}
