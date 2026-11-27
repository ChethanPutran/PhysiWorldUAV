"""Reliable imagination horizon, first-failure definition (thesis): the number of leading steps whose
position error stays <= threshold. A window is reliable at k only if every step up to k is."""
from __future__ import annotations

import torch

from .metrics import position_error


def reliable_horizon(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """(B,) integer number of reliable steps (0 … K)."""
    ok = (position_error(pred, target) <= threshold).float()
    return torch.cumprod(ok, dim=1).sum(dim=1)
