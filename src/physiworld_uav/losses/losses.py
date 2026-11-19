"""Thesis loss (docs/specs/model_specification.md §5), quaternion attitude.

    L_state = λ_p L_p + λ_v L_v + λ_R L_R + λ_ω L_ω
    L_roll  = Σ_k γ^(k-1) L_state(ŝ_{t+k}, s_{t+k})
    L       = L_state + λ_roll · L_roll

L_R = d_q(q̂, q)² with d_q the quaternion geodesic angle (rad). States are 13-dim: p|v|q|ω_B.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from physiworld_uav.models import quat

TERMS = ("p", "v", "R", "w")


@dataclass
class LossWeights:
    p: float = 1.0
    v: float = 1.0
    R: float = 1.0
    w: float = 1.0
    gamma: float = 0.99
    lambda_roll: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def state_terms(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-sample terms, shape ``pred.shape[:-1]``."""
    return {
        "p": (pred[..., 0:3] - target[..., 0:3]).square().sum(-1),
        "v": (pred[..., 3:6] - target[..., 3:6]).square().sum(-1),
        "R": quat.geodesic_angle(pred[..., 6:10], target[..., 6:10]).square(),
        "w": (pred[..., 10:13] - target[..., 10:13]).square().sum(-1),
    }


def state_loss(pred: torch.Tensor, target: torch.Tensor, w: LossWeights) -> torch.Tensor:
    t = state_terms(pred, target)
    return w.p * t["p"] + w.v * t["v"] + w.R * t["R"] + w.w * t["w"]       # per-sample


def rollout_discounts(k: int, gamma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    return gamma ** torch.arange(k, device=device, dtype=dtype)


def total_loss(out: dict, batch: dict, horizon: int, w: LossWeights) -> dict[str, torch.Tensor]:
    """``out``: model output (``dec`` (B,H,13) decoded history, ``pred`` (B,K,13)); returns the loss components.

    The observation-phase term supervises the decoder at every history step (absent for models
    without a decoder, i.e. when ``out`` has no ``dec``). λ_roll is divided by Σγ^(k-1) so the gradient
    scale does not grow with the curriculum horizon (decision D5).
    """
    target = batch["future_states"][:, :horizon]
    pred = out["pred"][:, :horizon]
    disc = rollout_discounts(horizon, w.gamma, pred.device, pred.dtype)
    per_k = state_loss(pred, target, w)                                    # (B,K)
    l_roll = (per_k * disc).sum(1).mean()
    parts = {"roll": l_roll}
    l_state = pred.new_zeros(())
    if "dec" in out and out["dec"] is not None:
        l_state = state_loss(out["dec"], batch["state_history"], w).mean()
    parts["state"] = l_state
    lam = w.lambda_roll / disc.sum()
    parts["total"] = l_state + lam * l_roll
    for k, v in state_terms(pred, target).items():                         # diagnostics, not in the loss
        parts[f"rollout_{k}"] = v.mean().detach()
    return parts


@torch.no_grad()
def fit_loss_weights(dataset, horizon: int, n: int = 4096, gamma: float = 0.99, lambda_roll: float = 1.0,
                     seed: int = 0) -> LossWeights:
    """λ_p … λ_ω = 1 / (mean term value of the hold-last-state baseline on training windows), so that no
    term dominates (a way of choosing λ, not a change to the loss)."""
    from physiworld_uav.models.world_model import HoldLastState
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[: min(n, len(dataset))].tolist()
    items = [dataset[i] for i in idx]
    batch = {k: torch.stack([it[k] for it in items]) for k in
             ("state_history", "obs_history", "action_history", "future_actions", "future_states")}
    pred = HoldLastState().predict(batch, horizon)
    terms = state_terms(pred, batch["future_states"][:, :horizon])
    inv = {k: float(1.0 / max(v.mean().item(), 1e-8)) for k, v in terms.items()}
    return LossWeights(p=inv["p"], v=inv["v"], R=inv["R"], w=inv["w"], gamma=gamma, lambda_roll=lambda_roll)
