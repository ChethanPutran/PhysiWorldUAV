"""Evaluate a model on a ModelReadyDataset: autoregressive rollout from the window end, errors by horizon.

Primary metric = RMSE over windows, sqrt(mean_w ‖e_k‖²) (keys ``pos_rmse_m``, ``vel_rmse_mps``, ``att_rmse_deg``,
``omega_rmse_radps``). Secondary = mean Euclidean error, mean_w ‖e_k‖ (``*_mean_*``). Attitude error is the geodesic angle.

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
    from physiworld_uav.data.dataset import ORACLE_KEYS
    return {k: batch[k].to(device) for k in BATCH_KEYS + tuple(k for k in ORACLE_KEYS + ("future_obs", "vehicle") if k in batch)}


@torch.no_grad()
def collect_errors(model, dataset, *, horizon: int, device="cpu", batch_size: int = 512, initial_state: str = "given",
                   anchor=None, threshold: float = 0.5) -> dict:
    """Per-window error arrays (N, horizon) of position [m], attitude [deg], velocity [m/s], ω [rad/s], the reliable horizon
    and the episode id of every window. ``anchor`` (``models.world_model.Anchor``) re-observes every K steps; the dataset must
    then provide ``future_obs`` unless the anchor resets the state only."""
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pe, ae, ve, we, rh, eids = [], [], [], [], [], []
    for batch in loader:
        b = _to(batch, device)
        pred = model.predict(b, horizon, initial_state) if anchor is None else model.predict(b, horizon, initial_state, anchor)
        tgt = b["future_states"][:, :horizon]
        pe.append(position_error(pred, tgt).cpu()); ae.append(attitude_error_deg(pred, tgt).cpu())
        ve.append(velocity_error(pred, tgt).cpu()); we.append(angular_velocity_error(pred, tgt).cpu())
        rh.append(reliable_horizon(pred, tgt, threshold).cpu()); eids += list(batch["episode_id"])
    pe, ae, ve, we, rh = (torch.cat(x).numpy() for x in (pe, ae, ve, we, rh))
    return {"pe": pe, "ae": ae, "ve": ve, "we": we, "rh": rh, "eids": np.array(eids)}


@torch.no_grad()
def evaluate(model, dataset, *, horizon: int, device="cpu", batch_size: int = 512, initial_state: str = "given",
             ks: Sequence[int] = (1, 5, 10, 20), threshold: float = 0.5, n_boot: int = 1000, seed: int = 0, anchor=None) -> dict:
    """Returns a JSON-able dict with per-horizon RMSE / mean (and p90) of position/attitude/velocity/ω errors,
    the mean reliable horizon, window and episode counts and a bootstrap CI (over episodes) of the
    per-episode position RMSE at the last k (mean over episodes)."""
    if len(dataset) == 0:
        return {"n_windows": 0, "n_episodes": 0}
    ks = [k for k in ks if k <= horizon]
    E = collect_errors(model, dataset, horizon=horizon, device=device, batch_size=batch_size, initial_state=initial_state,
                       anchor=anchor, threshold=threshold)
    pe, ae, ve, we, rh, eids = E["pe"], E["ae"], E["ve"], E["we"], E["rh"], E["eids"]
    out: dict = {"n_windows": int(len(pe)), "n_episodes": int(len(set(eids))), "horizon": horizon,
                 "initial_state": initial_state, "reliable_threshold_m": threshold,
                 "reliable_horizon_mean": float(rh.mean()), "finite": bool(np.isfinite(pe).all())}
    for stem, unit, arr in (("pos", "m", pe), ("att", "deg", ae), ("vel", "mps", ve), ("omega", "radps", we)):
        out[f"{stem}_rmse_{unit}"] = {str(k): float(np.sqrt((arr[:, k - 1] ** 2).mean())) for k in ks}
        out[f"{stem}_mean_{unit}"] = {str(k): float(arr[:, k - 1].mean()) for k in ks}
    out["pos_p90_m"] = {str(k): float(np.percentile(pe[:, k - 1], 90)) for k in ks}
    k_last = ks[-1]
    ep_ids = sorted(set(eids))
    per_ep = np.array([np.sqrt((pe[eids == e, k_last - 1] ** 2).mean()) for e in ep_ids])    # RMSE within each episode
    out["pos_rmse_m_episode_mean_last_k"] = float(per_ep.mean())
    if len(per_ep) > 1 and n_boot:
        rng = np.random.default_rng(seed)
        boots = [per_ep[rng.integers(0, len(per_ep), len(per_ep))].mean() for _ in range(n_boot)]
        out["pos_rmse_m_last_k_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out
