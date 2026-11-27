"""Thesis loss (docs/specs/model_specification.md §5), quaternion attitude.

    L_state = λ_p L_p + λ_v L_v + λ_R L_R + λ_ω L_ω
    L_roll  = Σ_k γ^(k-1) L_state(ŝ_{t+k}, s_{t+k})
    L       = λ_dec · L_dec + λ_roll · L_roll + λ_d · mean ‖d_θ‖²      (APHYNITY-style; only for models with a residual)

The one-step term is the k = 1 term of L_roll (there is no separate one-step loss). L_dec = L_state(D(h_j), s_j) over the
history. λ_roll is divided by Σ_k γ^(k−1), so the rollout gradient scale does not grow with the curriculum horizon.

L_R = d_q(q̂, q)² with d_q the quaternion geodesic angle (rad). States are 13-dim: p|v|q|ω_B.
"""
from __future__ import annotations

import math
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
    dec: float = 1.0           # λ_dec: weight of the decoder reconstruction term (1.0 = the behaviour before this knob existed)
    resid: float = 1e-3        # APHYNITY-style minimal-norm penalty on d_θ (dimensionless residual output: the network output
                               # BEFORE the per-channel scale, so it does not depend on data statistics); 0 disables
    zero: float = 0.0          # λ₀: soft penalty ‖d_θ(h = 0, ŝ, u)‖² (option b of model_specification §3.4); 0 = off

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
    parts["total"] = w.dec * l_state + lam * l_roll
    if out.get("resid_sq") is not None:
        l_res = out["resid_sq"][:, :horizon].mean()
        parts["resid"] = l_res.detach()
        if w.resid:
            parts["total"] = parts["total"] + w.resid * l_res
    if out.get("resid0_sq") is not None:
        l_zero = out["resid0_sq"][:, :horizon].mean()
        parts["zero"] = l_zero.detach()
        if w.zero:
            parts["total"] = parts["total"] + w.zero * l_zero
    for k, v in state_terms(pred, target).items():                         # diagnostics, not in the loss
        parts[f"rollout_{k}"] = v.mean().detach()
    return parts


# Physical resolution below which a baseline error is treated as noise (squared units of each term): 1 cm, 1 cm/s,
# 0.5 deg, 0.01 rad/s. PROPOSED values: they only stop a near-perfect baseline term (e.g. attitude in hover) from
# producing an enormous weight.
PHYSICAL_RESOLUTION = {"p": 0.01, "v": 0.01, "R": math.radians(0.5), "w": 0.01}


@torch.no_grad()
def fit_loss_weights(dataset, horizon: int, n: int = 4096, gamma: float = 0.99, lambda_roll: float = 1.0,
                     seed: int = 0, resid: float = 1e-3, dec: float = 1.0, zero: float = 0.0, resolution: dict | None = None,
                     return_info: bool = False):
    """λ_p … λ_ω = 1 / max(mean term value of the hold-last-state baseline on training windows, resolution²), so that no
    term dominates and a near-zero baseline cannot give a huge weight (a way of choosing λ, not a change to the loss).
    ``return_info`` also returns, per term, the baseline value, the floor and whether the floor was active."""
    from physiworld_uav.models.world_model import HoldLastState
    res = {**PHYSICAL_RESOLUTION, **(resolution or {})}
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[: min(n, len(dataset))].tolist()
    items = [dataset[i] for i in idx]
    batch = {k: torch.stack([it[k] for it in items]) for k in
             ("state_history", "obs_history", "action_history", "future_actions", "future_states")}
    pred = HoldLastState().predict(batch, horizon)
    terms = state_terms(pred, batch["future_states"][:, :horizon])
    base = {k: float(v.mean().item()) for k, v in terms.items()}
    floor = {k: float(res[k]) ** 2 for k in terms}
    inv = {k: 1.0 / max(base[k], floor[k]) for k in terms}
    lw = LossWeights(p=inv["p"], v=inv["v"], R=inv["R"], w=inv["w"], gamma=gamma, lambda_roll=lambda_roll, resid=resid, dec=dec, zero=zero)
    if not return_info:
        return lw
    return lw, {k: {"baseline": base[k], "floor": floor[k], "floor_active": base[k] < floor[k], "weight": inv[k]} for k in terms}


def weight_table(info: dict) -> str:
    """Per-component weights as a text table (printed before training)."""
    rows = ["term | hold-last baseline | physical floor | floor active | weight",
            "-----|--------------------|----------------|--------------|-------"]
    rows += [f"{k:4s} | {v['baseline']:18.4g} | {v['floor']:14.4g} | {str(v['floor_active']):12s} | {v['weight']:.4g}" for k, v in info.items()]
    return "\n".join(rows)
