"""Evaluate a model on a ModelReadyDataset: autoregressive rollout from the window end, errors by horizon.

Aggregation is per window for the horizon curves and per **episode** for the uncertainty (windows of one
episode overlap and are not independent): the bootstrap resamples episodes.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import angular_velocity_error, attitude_error_deg, position_error, velocity_error
from .reliable_horizon import reliable_horizon

BATCH_KEYS = ("state_history", "obs_history", "action_history", "future_actions", "future_states")


def _to(batch: dict, device) -> dict:
    return {k: batch[k].to(device) for k in BATCH_KEYS}


@torch.no_grad()
def evaluate(model, dataset, *, horizon: int, device="cpu", batch_size: int = 512, initial_state: str = "given",
             ks: Sequence[int] = (1, 5, 10, 20), threshold: float = 0.5, n_boot: int = 1000, seed: int = 0) -> dict:
    """Returns a JSON-able dict with per-horizon means (and p90) of position/attitude/velocity/ω errors,
    the mean reliable horizon, window and episode counts and a bootstrap CI (over episodes) of the
    position error at the last k."""
    if len(dataset) == 0:
        return {"n_windows": 0, "n_episodes": 0}
    model.eval().to(device)
    ks = [k for k in ks if k <= horizon]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pe, ae, ve, we, rh, eids = [], [], [], [], [], []
    for batch in loader:
        b = _to(batch, device)
        pred = model.predict(b, horizon, initial_state)
        tgt = b["future_states"][:, :horizon]
        pe.append(position_error(pred, tgt).cpu()); ae.append(attitude_error_deg(pred, tgt).cpu())
        ve.append(velocity_error(pred, tgt).cpu()); we.append(angular_velocity_error(pred, tgt).cpu())
        rh.append(reliable_horizon(pred, tgt, threshold).cpu()); eids += list(batch["episode_id"])
    pe, ae, ve, we, rh = (torch.cat(x).numpy() for x in (pe, ae, ve, we, rh))
    eids = np.array(eids)
    out: dict = {"n_windows": int(len(pe)), "n_episodes": int(len(set(eids))), "horizon": horizon,
                 "initial_state": initial_state, "reliable_threshold_m": threshold,
                 "reliable_horizon_mean": float(rh.mean()), "finite": bool(np.isfinite(pe).all())}
    for name, arr in (("pos_m", pe), ("att_deg", ae), ("vel_mps", ve), ("omega_radps", we)):
        out[name] = {str(k): float(arr[:, k - 1].mean()) for k in ks}
    out["pos_m_p90"] = {str(k): float(np.percentile(pe[:, k - 1], 90)) for k in ks}
    k_last = ks[-1]
    ep_ids = sorted(set(eids))
    per_ep = np.array([pe[eids == e, k_last - 1].mean() for e in ep_ids])
    out["pos_m_episode_mean_last_k"] = float(per_ep.mean())
    if len(per_ep) > 1 and n_boot:
        rng = np.random.default_rng(seed)
        boots = [per_ep[rng.integers(0, len(per_ep), len(per_ep))].mean() for _ in range(n_boot)]
        out["pos_m_last_k_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out
