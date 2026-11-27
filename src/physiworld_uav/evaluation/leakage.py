"""Shuffled-u test (``.idea/dataleakage``): does a model lean on the controller's wind information in ``u``?

In closed-loop (Mode A) counterfactual groups the siblings fly the same maneuver under a different wind magnitude, so the
controller commands different rotor speeds. The test replaces ``action_history`` and ``future_actions`` of a window by the
actions of the sibling with the most different wind at the same time index, and keeps the target of the original window.
``u`` legitimately drives the dynamics, so some error increase is expected for every model, B1 (physics only) included: read
the jump of B3/B2/B4 **relative to B1**. Mode B (fixed-action) groups are skipped: their siblings share exactly the same u.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from physiworld_uav.data.dataset import ModelReadyDataset
from . import regimes as R


def partner_map(episodes, min_peak_diff: float = 0.1) -> dict[str, object]:
    """episode_id → sibling (same Mode A wind-magnitude group) with the most different peak wind (≥ ``min_peak_diff`` m/s)."""
    groups: dict[str, list] = {}
    for ep in episodes:
        if R.cf_mode(ep) == "closed_loop" and R.cf_variable(ep) == "wind_magnitude":
            groups.setdefault(R.cf_group(ep), []).append(ep)
    out = {}
    for eps in groups.values():
        for ep in eps:
            cand = [o for o in eps if o is not ep and abs(R.peak_wind(o) - R.peak_wind(ep)) >= min_peak_diff]
            if cand:
                out[ep.episode_id] = max(cand, key=lambda o: abs(R.peak_wind(o) - R.peak_wind(ep)))
    return out


class ActionSwap(Dataset):
    """The windows of ``ds`` that have a partner long enough, with (``swap=True``) or without the partner's actions."""

    def __init__(self, ds: ModelReadyDataset, partners: dict, swap: bool) -> None:
        self.ds, self.swap = ds, swap
        H, P = ds.cfg.history, ds.cfg.horizon
        self.index = []
        for i in range(len(ds)):
            ep, t = ds.window(i)
            q = partners.get(ep.episode_id)
            if q is not None and len(q.actions) >= t + P:
                self.index.append((i, q))
        self.H, self.P = H, P

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, j: int) -> dict:
        i, q = self.index[j]
        item = self.ds[i]
        if self.swap:
            _, t = self.ds.window(i)
            item = dict(item)
            item["action_history"] = torch.from_numpy(np.asarray(q.actions[t - self.H: t]))
            item["future_actions"] = torch.from_numpy(np.asarray(q.actions[t: t + self.P]))
        return item
