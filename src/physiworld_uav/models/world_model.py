"""Latent world models and baselines (thesis B0–B4; B5, B6 share these parts).

Common interface (state layout p|v|q|ω_B, 13-dim; positions are handled relative to the
first state of the history window, so the nets are translation invariant)::

    model.predict(batch, horizon, initial_state="given") -> (B, K, 13) predicted future states
    model(batch, horizon, initial_state)                 -> dict(dec=(B,H,13) decoded history,
                                                                 pred=(B,K,13))

Index alignment (``docs/specs/model_specification.md`` §3.1): the encoder step j uses
(o_j, u_j) and the recurrence uses u_{j-1}; ``u_j`` for the last history step is
``future_actions[0]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from . import quat
from .physics import QuadrotorPhysics, rk4_step

STAT_KEYS = ("imu_mu", "imu_sd", "act_mu", "act_sd", "v_mu", "v_sd", "w_mu", "w_sd", "p_sd",
             "dp_sd", "dv_sd", "dth_sd", "dw_sd")


# ------------------------------------------------------------------ statistics
def fit_stats(episodes: Sequence) -> dict[str, list[float]]:
    """Normalisation statistics from **training** episodes only (saved with the model)."""
    S = np.concatenate([e.states for e in episodes]).astype(np.float64)
    O = np.concatenate([e.obs for e in episodes]).astype(np.float64)
    A = np.concatenate([e.actions for e in episodes]).astype(np.float64)
    rel = np.concatenate([e.states[:, 0:3] - e.states[:1, 0:3] for e in episodes])
    s0 = np.concatenate([e.states[:-1] for e in episodes]); s1 = np.concatenate([e.states[1:] for e in episodes])
    dth = quat.to_rotation_vector(quat.mul(torch.tensor(s1[:, 6:10]), quat.conj(torch.tensor(s0[:, 6:10])))).numpy()
    sd = lambda x: np.maximum(x.std(0), 1e-6).tolist()  # noqa: E731
    return dict(imu_mu=O.mean(0).tolist(), imu_sd=sd(O), act_mu=A.mean(0).tolist(), act_sd=sd(A),
                v_mu=S[:, 3:6].mean(0).tolist(), v_sd=sd(S[:, 3:6]), w_mu=S[:, 10:13].mean(0).tolist(),
                w_sd=sd(S[:, 10:13]), p_sd=sd(rel), dp_sd=sd(s1[:, 0:3] - s0[:, 0:3]),
                dv_sd=sd(s1[:, 3:6] - s0[:, 3:6]), dth_sd=sd(dth), dw_sd=sd(s1[:, 10:13] - s0[:, 10:13]))


def _mlp(i: int, hidden: Sequence[int], o: int) -> nn.Sequential:
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ helpers
def relative_frame(batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference point = position at the first history step (shape (B,1,3))."""
    p_ref = batch["state_history"][:, :1, 0:3]
    return p_ref, p_ref


def _shift(states: torch.Tensor, p_ref: torch.Tensor, sign: float) -> torch.Tensor:
    out = states.clone()
    out[..., 0:3] = out[..., 0:3] + sign * p_ref
    return out


# ------------------------------------------------------------------ B0
class HoldLastState(nn.Module):
    """B0 (extra reference, not in the thesis table): zero-order hold of the last observed state."""
    name = "b0"

    def predict(self, batch, horizon: int, initial_state: str = "given") -> torch.Tensor:
        return batch["state_history"][:, -1:].expand(-1, horizon, -1).clone()


# ------------------------------------------------------------------ B1
class PhysicsOnly(nn.Module):
    """B1: ṡ = f_phys(ŝ, u), RK4, nominal parameters, no wind. Nothing is learned. Starts from the given state."""
    name = "b1"

    def __init__(self, physics: QuadrotorPhysics, dt: float = 0.02) -> None:
        super().__init__()
        self.physics, self.dt = physics, float(dt)

    @torch.no_grad()
    def predict(self, batch, horizon: int, initial_state: str = "given") -> torch.Tensor:
        if initial_state != "given":
            raise ValueError("B1 has no encoder/decoder: initial_state must be 'given'")
        s, outs = self.physics.project(batch["state_history"][:, -1]), []
        for k in range(horizon):
            u = batch["future_actions"][:, k]
            s = rk4_step(lambda x: self.physics.derivative(x, u), s, self.dt, self.physics.project)
            outs.append(s)
        return torch.stack(outs, dim=1)


# ------------------------------------------------------------------ latent models (B2, B3)
@dataclass
class ModelConfig:
    latent: int = 128          # h
    embed: int = 64            # e_t and φ(s)
    hidden: int = 256          # encoder / decoder / φ width
    trans_hidden: int = 0      # B2 transition width; 0 = auto-match the parameter count of B3 (±10 %, thesis Stage 2)
    resid_hidden: int = 256    # B3 residual width; B4 width is auto-matched to B3 when ``b4_hidden`` is 0
    b4_hidden: int = 0
    residual_channels: str = "all"   # "all" (thesis-exact: d_θ acts on the whole ṡ) | "acc_only" (v̇, ω̇ only; ablation)


class _LatentBase(nn.Module):
    """Shared parts of B2/B3 (thesis equations 23, 24, 25, 26):

    * ``e_t = E(IMU_t, u_t)``
    * ``h_t = F(h_{t-1}, e_t, u_{t-1})`` (GRU cell, input [e, u_prev])
    * ``ŝ_t = D(h_t)`` (positions relative to the window start; quaternion normalised)
    * coupling: ``h_{t+1} = F(h_t, φ(s̃_{t+1}), u_t)`` with the same cell

    Subclasses define ``transition(s, u, h) -> s̃_{t+1}`` (equation 5)."""

    def __init__(self, stats: Mapping[str, Sequence[float]], cfg: ModelConfig, dt: float) -> None:
        super().__init__()
        self.cfg, self.dt = cfg, float(dt)
        c = cfg
        for k in STAT_KEYS:
            self.register_buffer(k, torch.tensor(stats[k], dtype=torch.float32))
        self.enc = _mlp(6 + 4, [c.hidden // 2], c.embed)
        self.cell = nn.GRUCell(c.embed + 4, c.latent)
        self.dec = _mlp(c.latent, [c.hidden], 13)
        self.phi = _mlp(3 + 4 + 3, [c.hidden // 2], c.embed)
        with torch.no_grad():
            self.dec[-1].bias.zero_()

    # ---- normalisation
    def _u(self, u): return (u - self.act_mu) / self.act_sd
    def _imu(self, o): return (o - self.imu_mu) / self.imu_sd

    def _state_in(self, s: torch.Tensor) -> torch.Tensor:        # position-free state features (10)
        return torch.cat([(s[..., 3:6] - self.v_mu) / self.v_sd, s[..., 6:10], (s[..., 10:13] - self.w_mu) / self.w_sd], -1)

    # ---- E, F
    def observe(self, obs_hist: torch.Tensor, action_hist: torch.Tensor, first_future_action: torch.Tensor) -> torch.Tensor:
        """Latent states h_j for every history step j (B,H,latent)."""
        u_cur = torch.cat([action_hist[:, 1:], first_future_action[:, None]], dim=1)     # u_{t-H+1..t}
        h = obs_hist.new_zeros(obs_hist.shape[0], self.cfg.latent)
        hs = []
        for j in range(obs_hist.shape[1]):
            e = self.enc(torch.cat([self._imu(obs_hist[:, j]), self._u(u_cur[:, j])], -1))
            h = self.cell(torch.cat([e, self._u(action_hist[:, j])], -1), h)
            hs.append(h)
        return torch.stack(hs, dim=1)

    # ---- D
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        y = self.dec(h)
        q = quat.normalize(y[..., 6:10] + torch.tensor([1.0, 0, 0, 0], device=y.device))
        return torch.cat([y[..., 0:3] * self.p_sd, y[..., 3:6] * self.v_sd + self.v_mu, q,
                          y[..., 10:13] * self.w_sd + self.w_mu], -1)

    def couple(self, h: torch.Tensor, s_next: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.cell(torch.cat([self.phi(self._state_in(s_next)), self._u(u)], -1), h)

    def transition(self, s: torch.Tensor, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:   # pragma: no cover
        raise NotImplementedError

    def forward(self, batch: Mapping[str, torch.Tensor], horizon: int, initial_state: str = "given") -> dict:
        p_ref, _ = relative_frame(batch)
        hs = self.observe(batch["obs_history"], batch["action_history"], batch["future_actions"][:, 0])
        dec = self.decode(hs)                                              # (B,H,13), relative positions
        if initial_state == "given":
            s = _shift(batch["state_history"][:, -1], p_ref[:, 0], -1.0)
        elif initial_state == "decoder":
            s = dec[:, -1]
        else:
            raise ValueError("initial_state must be 'given' or 'decoder'")
        h, outs = hs[:, -1], []
        for k in range(horizon):
            u = batch["future_actions"][:, k]
            s = self.transition(s, u, h)
            h = self.couple(h, s, u)
            outs.append(s)
        return dict(dec=_shift(dec, p_ref, 1.0), pred=_shift(torch.stack(outs, 1), p_ref, 1.0))

    def predict(self, batch, horizon: int, initial_state: str = "given") -> torch.Tensor:
        return self(batch, horizon, initial_state)["pred"]


class PureNeuralLatent(_LatentBase):
    """B2: pure neural latent dynamics (thesis Stage 2). Learned transition
    ``ŝ_{t+1} = F_pure(ŝ_t, u_t, h_t)`` in place of f_phys + d_θ: increments (Δp, Δv, rotation-vector Δθ composed
    with q, Δω), so the quaternion stays unit. No f_phys."""
    name = "b2"

    def __init__(self, stats, cfg: ModelConfig | None = None, dt: float = 0.02) -> None:
        cfg = cfg or ModelConfig()
        super().__init__(stats, cfg, dt)
        w = cfg.trans_hidden or _auto_width("b2", stats, cfg, dt)
        self.step_net = _mlp(cfg.latent + 10 + 4, [w, w], 12)
        with torch.no_grad():
            self.step_net[-1].weight.mul_(0.05); self.step_net[-1].bias.zero_()

    def transition(self, s, u, h):
        d = self.step_net(torch.cat([h, self._state_in(s), self._u(u)], -1))
        dp, dv, dth, dw = (d[..., 0:3] * self.dp_sd, d[..., 3:6] * self.dv_sd, d[..., 6:9] * self.dth_sd, d[..., 9:12] * self.dw_sd)
        q = quat.normalize(quat.mul(quat.from_rotation_vector(dth), s[..., 6:10]))
        return torch.cat([s[..., 0:3] + dp, s[..., 3:6] + dv, q, s[..., 10:13] + dw], -1)


def _residual_scale(stats, dt: float) -> torch.Tensor:
    """Typical derivative magnitudes [ṗ, v̇, attitude rate, ω̇] used to scale the residual output."""
    return torch.cat([torch.tensor(stats[k], dtype=torch.float32) for k in ("dp_sd", "dv_sd", "dth_sd", "dw_sd")]) / dt


def _residual_mask(channels: str) -> torch.Tensor:
    if channels == "all":
        return torch.ones(12)
    if channels == "acc_only":                        # keep v̇ and ω̇; ṗ and attitude-rate corrections are zero
        m = torch.zeros(12); m[3:6] = 1; m[9:12] = 1
        return m
    raise ValueError("residual_channels must be 'all' or 'acc_only'")


class PhysicsResidualLatent(_LatentBase):
    """B3, the proposed model (thesis): ``s̃_{t+1} = RK4(ŝ_t, f_phys(ŝ_t,u_t) + d_θ(h_t, ŝ_t, u_t), Δt)`` with the
    latent memory ``h`` and the state→latent coupling of equation 6. ``d_θ`` acts in the tangent space of the state
    (ṗ, v̇, attitude rate δ, ω̇); the quaternion kinematics use ω_B + δ so ``|q| = 1`` is preserved. The residual is
    re-evaluated at each RK4 stage with ``h`` and ``u`` held over the step."""
    name = "b3"

    def __init__(self, stats, cfg: ModelConfig | None = None, dt: float = 0.02, physics: QuadrotorPhysics | None = None) -> None:
        cfg = cfg or ModelConfig()
        super().__init__(stats, cfg, dt)
        self.physics = physics or QuadrotorPhysics.from_yaml()
        w = cfg.resid_hidden
        self.resid = _mlp(cfg.latent + 10 + 4, [w, w], 12)
        with torch.no_grad():
            self.resid[-1].weight.mul_(0.05); self.resid[-1].bias.zero_()
        self.register_buffer("res_scale", _residual_scale(stats, dt))
        self.register_buffer("res_mask", _residual_mask(cfg.residual_channels))

    def residual(self, h: torch.Tensor, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.resid(torch.cat([h, self._state_in(s), self._u(u)], -1)) * self.res_scale * self.res_mask

    def transition(self, s, u, h):
        return rk4_step(lambda x: self.physics.derivative(x, u, self.residual(h, x, u)), s, self.dt, self.physics.project)


class PhysicsResidualNoLatent(nn.Module):
    """B4: physics + residual **without** latent memory: ``d_θ(s_t, u_t)``. No encoder, decoder or latent, so the
    start state is the given one. Width is auto-matched to B3's parameter count (thesis: isolate the memory)."""
    name = "b4"

    def __init__(self, stats, cfg: ModelConfig | None = None, dt: float = 0.02, physics: QuadrotorPhysics | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig(); self.dt = float(dt)
        for k in STAT_KEYS:
            self.register_buffer(k, torch.tensor(stats[k], dtype=torch.float32))
        self.physics = physics or QuadrotorPhysics.from_yaml()
        w = self.cfg.b4_hidden or _auto_width("b4", stats, self.cfg, dt)
        self.resid = _mlp(10 + 4, [w, w], 12)
        with torch.no_grad():
            self.resid[-1].weight.mul_(0.05); self.resid[-1].bias.zero_()
        self.register_buffer("res_scale", _residual_scale(stats, dt))
        self.register_buffer("res_mask", _residual_mask(self.cfg.residual_channels))

    def _u(self, u): return (u - self.act_mu) / self.act_sd

    def _state_in(self, s): return torch.cat([(s[..., 3:6] - self.v_mu) / self.v_sd, s[..., 6:10], (s[..., 10:13] - self.w_mu) / self.w_sd], -1)

    def residual(self, s, u):
        return self.resid(torch.cat([self._state_in(s), self._u(u)], -1)) * self.res_scale * self.res_mask

    def forward(self, batch, horizon: int, initial_state: str = "given") -> dict:
        if initial_state != "given":
            raise ValueError("B4 has no encoder/decoder: initial_state must be 'given'")
        p_ref, _ = relative_frame(batch)
        s = self.physics.project(_shift(batch["state_history"][:, -1], p_ref[:, 0], -1.0))
        outs = []
        for k in range(horizon):
            u = batch["future_actions"][:, k]
            s = rk4_step(lambda x: self.physics.derivative(x, u, self.residual(x, u)), s, self.dt, self.physics.project)
            outs.append(s)
        return dict(pred=_shift(torch.stack(outs, 1), p_ref, 1.0), dec=None)

    def predict(self, batch, horizon: int, initial_state: str = "given") -> torch.Tensor:
        return self(batch, horizon, initial_state)["pred"]


def n_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _auto_width(kind: str, stats, cfg: ModelConfig, dt: float) -> int:
    """Hidden width (two hidden layers) for B2's transition / B4's residual so that the parameter count is as
    close as possible to B3's (thesis: 'approximately comparable parameter capacity')."""
    target = n_parameters(PhysicsResidualLatent(stats, cfg, dt))
    base = n_parameters(_LatentBase(stats, cfg, dt)) if kind == "b2" else 0
    best, best_gap = 16, float("inf")
    for w in range(8, 1025, 2):
        if kind == "b2":
            n = base + (cfg.latent + 14) * w + w + w * w + w + w * 12 + 12
        else:   # b4
            n = (14 * w + w) + (w * w + w) + (w * 12 + 12)
        if abs(n - target) < best_gap:
            best, best_gap = w, abs(n - target)
    return best


def build_model(kind: str, *, stats=None, physics: QuadrotorPhysics | None = None, dt: float = 0.02,
                cfg: ModelConfig | None = None) -> nn.Module:
    kind = kind.lower()
    if kind == "b0":
        return HoldLastState()
    if kind == "b1":
        return PhysicsOnly(physics or QuadrotorPhysics.from_yaml(), dt)
    if kind in ("b2", "b3", "b4"):
        if stats is None:
            raise ValueError(f"{kind} needs normalisation statistics (fit_stats on the training episodes)")
        cls = {"b2": PureNeuralLatent, "b3": PhysicsResidualLatent, "b4": PhysicsResidualNoLatent}[kind]
        return cls(stats, cfg, dt) if kind == "b2" else cls(stats, cfg, dt, physics)
    raise ValueError(f"unknown model {kind!r}; implemented: b0, b1, b2, b3, b4 (B5, B6 not implemented yet)")
