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

import warnings

import numpy as np
import torch
import torch.nn as nn

from . import quat
from .physics import G as GRAVITY, QuadrotorPhysics, rk4_step

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


ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "elu": nn.ELU}


def _mlp(i: int, hidden: Sequence[int], o: int, act: str = "silu") -> nn.Sequential:
    if act not in ACTIVATIONS:
        raise ValueError(f"unknown activation {act!r}; valid: {sorted(ACTIVATIONS)}")
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), ACTIVATIONS[act]()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def mlp_params(i: int, hidden: Sequence[int], o: int) -> int:
    """Parameter count of ``_mlp(i, hidden, o)`` (weights and biases)."""
    n, d = 0, i
    for h in list(hidden) + [o]:
        n += d * h + h
        d = h
    return n


# ------------------------------------------------------------------ re-observation (evaluation only)
@dataclass(frozen=True)
class Anchor:
    """Re-observe the vehicle every ``every`` rollout steps (receding-horizon use), instead of pure open loop.

    Latent models (B2, B3, B5) always update the latent with the REAL IMU of that step (needs ``batch["future_obs"]``).
    ``state=False``: the physical state keeps free-running. ``state=True``: additionally the physical state is reset to the
    true state (a perfect state estimator). Models without a latent (B1, B4, B6a) have no IMU input: only ``state=True`` is
    meaningful for them and needs no future IMU."""
    every: int
    state: bool = False

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError("every must be >= 1")

    def fires(self, k: int, horizon: int) -> bool:
        """After rollout step k (0-based) the next input exists only if k + 1 < horizon."""
        return (k + 1) % self.every == 0 and k < horizon - 1


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

    def predict(self, batch, horizon: int, initial_state: str = "given", anchor: Anchor | None = None) -> torch.Tensor:
        return batch["state_history"][:, -1:].expand(-1, horizon, -1).clone()

    def forward(self, batch, horizon: int, initial_state: str = "given", anchor: Anchor | None = None) -> dict:
        return dict(pred=self.predict(batch, horizon, initial_state, anchor), dec=None)


# ------------------------------------------------------------------ B1
class PhysicsOnly(nn.Module):
    """B1: ṡ = f_phys(ŝ, u), RK4, nominal parameters, no wind. Nothing is learned. Starts from the given state."""
    name = "b1"

    def __init__(self, physics: QuadrotorPhysics, dt: float = 0.02) -> None:
        super().__init__()
        self.physics, self.dt = physics, float(dt)

    @torch.no_grad()
    def predict(self, batch, horizon: int, initial_state: str = "given", anchor: Anchor | None = None) -> torch.Tensor:
        if initial_state != "given":
            raise ValueError("B1 has no encoder/decoder: initial_state must be 'given'")
        s, outs = self.physics.project(batch["state_history"][:, -1]), []
        for k in range(horizon):
            u = batch["future_actions"][:, k]
            s = rk4_step(lambda x: self.physics.derivative(x, u), s, self.dt, self.physics.project)
            outs.append(s)
            if anchor is not None and anchor.state and anchor.fires(k, horizon):
                s = batch["future_states"][:, k]
        return torch.stack(outs, dim=1)

    def forward(self, batch, horizon: int, initial_state: str = "given", anchor: Anchor | None = None) -> dict:
        return dict(pred=self.predict(batch, horizon, initial_state, anchor), dec=None)


# ------------------------------------------------------------------ latent models (B2, B3)
@dataclass
class ModelConfig:
    """Every size of every model. All of it is set from YAML (``config/training/default.yaml`` and the files in
    ``config/model/``, see ``TrainConfig``); the defaults below reproduce the pilot architecture.

    Fixed by the data interface and therefore NOT configurable: IMU input 6, action 4, state features 10, state 13,
    residual / vector-field output 12.
    Depth/width rule of the dynamics networks (B2 transition, B3 / B6b residual, B4 residual, B5 field): ``*_depth``
    hidden layers of the stated width."""
    latent: int = 128          # h (GRU state size)
    embed: int = 64            # e_t and φ(s)
    hidden: int = 256          # base width: encoder hidden = hidden // 2 and decoder hidden = hidden unless set below
    enc_hidden: Sequence[int] | None = None   # encoder E hidden layers; None = [hidden // 2]
    dec_hidden: Sequence[int] | None = None   # decoder D hidden layers; None = [hidden]
    trans_depth: int = 2       # hidden layers of the B2 transition network
    resid_depth: int = 2       # hidden layers of the B3 / B6b residual network
    b4_depth: int = 2          # hidden layers of the B4 residual network
    field_depth: int = 2       # hidden layers of the B5 vector-field network
    activation: str = "silu"   # silu | relu | tanh | gelu | elu (all networks)
    # --- architectural-bias options against the action shortcut (model_specification.md §3.4); the defaults are the thesis model
    resid_uses_u: bool = True    # False (option a, "B3a"): d_θ(h, ŝ) has no action input (B3, B4, B6b); u still enters through f_phys
    shared_trunk: bool = False   # True (option c, "B3c"): d_θ reads the decoder's last hidden layer T(h) instead of h (B3, B6b)
    trans_hidden: int = 0      # B2 transition width; 0 = auto-match the parameter count of B3 (±10 %, thesis Stage 2)
    resid_hidden: int = 256    # B3 residual width; B4 width is auto-matched to B3 when ``b4_hidden`` is 0
    b4_hidden: int = 0
    b5_hidden: int = 0               # B5 field width; 0 = auto-match B3's parameter count (set >0 to run a wider B5)
    residual_channels: str = "acc_only"   # "acc_only" (v̇, ω̇ only; default, choice 5) | "all" (d_θ on the whole ṡ; ablation)
    enc_uses_u: bool = False         # False: E is IMU-only (u_t is chosen after step t); True: E(IMU_t, u_t) (ablation, B2 and B3)
    phi_noise: str = "none"          # IMU-noise preset (data/noise.py) whose WHITE part is added to the synthetic IMU of φ

    def __post_init__(self) -> None:
        for name in ("enc_hidden", "dec_hidden"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, [int(x) for x in v])
        for name in ("latent", "embed", "hidden"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"ModelConfig.{name} must be >= 1, got {getattr(self, name)}")
        for name in ("trans_depth", "resid_depth", "b4_depth", "field_depth"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"ModelConfig.{name} must be >= 0, got {getattr(self, name)}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}; valid: {sorted(ACTIVATIONS)}")


def enc_layers(c: ModelConfig) -> list[int]:
    return [int(x) for x in c.enc_hidden] if c.enc_hidden is not None else [c.hidden // 2]


def dec_layers(c: ModelConfig) -> list[int]:
    return [int(x) for x in c.dec_hidden] if c.dec_hidden is not None else [c.hidden]


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
        self.enc = _mlp(6 + (4 if c.enc_uses_u else 0), enc_layers(c), c.embed, c.activation)
        self.cell = nn.GRUCell(c.embed + 4, c.latent)
        self.dec = _mlp(c.latent, dec_layers(c), 13, c.activation)
        self.phi_noise = c.phi_noise     # mutable: trainer / evaluator set it to match the noise of the real IMU
        with torch.no_grad():
            self.dec[-1].bias.zero_()

    # ---- normalisation
    def _u(self, u): return (u - self.act_mu) / self.act_sd
    def _imu(self, o): return (o - self.imu_mu) / self.imu_sd

    def _state_in(self, s: torch.Tensor) -> torch.Tensor:        # position-free state features (10)
        return torch.cat([(s[..., 3:6] - self.v_mu) / self.v_sd, s[..., 6:10], (s[..., 10:13] - self.w_mu) / self.w_sd], -1)

    # ---- E, F
    def encode(self, imu: torch.Tensor, u_cur: torch.Tensor) -> torch.Tensor:
        """e = E(IMU) (default) or E(IMU, u_t) when ``cfg.enc_uses_u``. ``imu`` is already normalised."""
        return self.enc(torch.cat([imu, self._u(u_cur)], -1) if self.cfg.enc_uses_u else imu)

    def observe(self, obs_hist: torch.Tensor, action_hist: torch.Tensor, first_future_action: torch.Tensor) -> torch.Tensor:
        """Latent states h_j for every history step j (B,H,latent)."""
        u_cur = torch.cat([action_hist[:, 1:], first_future_action[:, None]], dim=1)     # u_{t-H+1..t}
        h = obs_hist.new_zeros(obs_hist.shape[0], self.cfg.latent)
        hs = []
        for j in range(obs_hist.shape[1]):
            e = self.encode(self._imu(obs_hist[:, j]), u_cur[:, j])
            h = self.cell(torch.cat([e, self._u(action_hist[:, j])], -1), h)
            hs.append(h)
        return torch.stack(hs, dim=1)

    # ---- D
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        y = self.dec(h)
        q = quat.normalize(y[..., 6:10] + torch.tensor([1.0, 0, 0, 0], device=y.device))
        return torch.cat([y[..., 0:3] * self.p_sd, y[..., 3:6] * self.v_sd + self.v_mu, q,
                          y[..., 10:13] * self.w_sd + self.w_mu], -1)

    # ---- φ: synthetic IMU from the predicted state, then the SAME encoder E and cell F as the observation phase
    def synth_imu(self, s_next: torch.Tensor, vdot: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
        """φ(·) = [R(q̂)ᵀ(v̇ − g), ω̂_B] (6) + white noise of the configured preset (no learned parameters).

        Matches the stored instantaneous IMU (specific force in the body frame, ω in the body frame)."""
        R = quat.to_rotation_matrix(quat.normalize(s_next[..., 6:10]))
        g = torch.tensor([0.0, 0.0, GRAVITY], device=s_next.device, dtype=s_next.dtype)
        acc = (R.transpose(-1, -2) @ (vdot + g).unsqueeze(-1)).squeeze(-1)
        imu = torch.cat([acc, s_next[..., 10:13]], -1)
        return imu + self._white_noise(imu, gen)

    def _white_noise(self, like: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
        from physiworld_uav.data.noise import ImuNoiseConfig
        c = ImuNoiseConfig.preset(self.phi_noise)
        if not (c.accel_white or c.gyro_white):
            return torch.zeros_like(like)
        sd = like.new_tensor([c.accel_white] * 3 + [c.gyro_white] * 3)
        return torch.randn(like.shape, device=like.device, dtype=like.dtype, generator=gen) * sd

    def couple(self, h, s_prev, s_next, u, u_next, extra_h, gen=None, ctx=None):
        """h_{t+1} = F(h_t, E(φ(s̃_{t+1}), u_{t+1}), u_t): the same recurrence as in ``observe``."""
        vdot = self.accel_next(s_prev, s_next, u_next, extra_h, ctx)
        e = self.encode(self._imu(self.synth_imu(s_next, vdot, gen)), u_next)
        return self.cell(torch.cat([e, self._u(u)], -1), h)

    def accel_next(self, s_prev, s_next, u_next, h, ctx=None) -> torch.Tensor:                 # pragma: no cover
        raise NotImplementedError

    def residual_sq(self, h, s, u, ctx=None) -> torch.Tensor | None:
        """Per-sample mean-square of the (dimensionless, masked) residual output, for the APHYNITY penalty.

        Dimensionless means: the network output BEFORE ``res_scale``, so ``λ_d`` does not depend on data statistics; a physical
        residual of 0.1 m/s² is a normalised output of about 0.02. It is evaluated at the state at the START of each step, not at
        the intermediate RK4 stages at which the dynamics evaluate ``d_θ`` (within one Δt of it; the difference is small)."""
        return None

    def residual_zero_sq(self, h, s, u, ctx=None) -> torch.Tensor | None:
        """``‖d_θ(h = 0, ŝ, u)‖²`` per sample (option b, ``lambda_zero``): the residual must vanish when the latent carries
        no information. None for models without a latent residual."""
        return None

    def transition(self, s: torch.Tensor, u: torch.Tensor, h: torch.Tensor, ctx=None) -> torch.Tensor:   # pragma: no cover
        raise NotImplementedError

    def forward(self, batch: Mapping[str, torch.Tensor], horizon: int, initial_state: str = "given",
                anchor: Anchor | None = None) -> dict:
        p_ref, _ = relative_frame(batch)
        hs = self.observe(batch["obs_history"], batch["action_history"], batch["future_actions"][:, 0])
        dec = self.decode(hs)                                              # (B,H,13), relative positions
        if initial_state == "given":
            s = _shift(batch["state_history"][:, -1], p_ref[:, 0], -1.0)
        elif initial_state == "decoder":
            s = dec[:, -1]
        else:
            raise ValueError("initial_state must be 'given' or 'decoder'")
        gen = None
        if not self.training:                         # reproducible evaluation when φ noise is on
            gen = torch.Generator(device=hs.device).manual_seed(0)
        h, outs, hts, rsq, rsq0 = hs[:, -1], [], [], [], []
        want_zero = self.training and getattr(self, "collect_zero_resid", False)     # set by the trainer when lambda_zero > 0
        for k in range(horizon):
            u = batch["future_actions"][:, k]
            hts.append(h)
            ctx = (batch, k)
            r = self.residual_sq(h, s, u, ctx)
            if r is not None:
                rsq.append(r)
                if want_zero:
                    rsq0.append(self.residual_zero_sq(h, s, u, ctx))
            s_prev = s
            s = self.transition(s, u, h, ctx)
            outs.append(s)
            if k < horizon - 1:        # the h after the last step is never used; u_{t+k+1} exists for k+1 < horizon
                if anchor is not None and anchor.fires(k, horizon):          # re-observation: real IMU of step t+k+1
                    if "future_obs" not in batch:
                        raise ValueError("re-observation feeds the real IMU into the latent: build the dataset with "
                                         "ModelReadyDataset(..., future_obs=True)")
                    e = self.encode(self._imu(batch["future_obs"][:, k]), batch["future_actions"][:, k + 1])
                    h = self.cell(torch.cat([e, self._u(u)], -1), h)
                    if anchor.state:
                        s = _shift(batch["future_states"][:, k], p_ref[:, 0], -1.0)
                else:
                    h = self.couple(h, s_prev, s, u, batch["future_actions"][:, k + 1], h, gen, ctx)
        out = dict(dec=_shift(dec, p_ref, 1.0), pred=_shift(torch.stack(outs, 1), p_ref, 1.0),
                   h_obs=hs[:, -1], h_traj=torch.stack(hts, 1))
        if rsq:
            out["resid_sq"] = torch.stack(rsq, 1)                       # (B,K)
        if rsq0 and rsq0[0] is not None:
            out["resid0_sq"] = torch.stack(rsq0, 1)
        return out

    def predict(self, batch, horizon: int, initial_state: str = "given", anchor: Anchor | None = None) -> torch.Tensor:
        return self(batch, horizon, initial_state, anchor)["pred"]


class PureNeuralLatent(_LatentBase):
    """B2: pure neural latent dynamics (thesis Stage 2). Learned transition
    ``ŝ_{t+1} = F_pure(ŝ_t, u_t, h_t)`` in place of f_phys + d_θ: increments (Δp, Δv, rotation-vector Δθ composed
    with q, Δω), so the quaternion stays unit. No f_phys."""
    name = "b2"

    def __init__(self, stats, cfg: ModelConfig | None = None, dt: float = 0.02) -> None:
        cfg = cfg or ModelConfig()
        super().__init__(stats, cfg, dt)
        w = cfg.trans_hidden or _auto_width("b2", stats, cfg, dt)
        self.step_net = _mlp(cfg.latent + 10 + 4, [w] * cfg.trans_depth, 12, cfg.activation)
        with torch.no_grad():
            self.step_net[-1].weight.mul_(0.05); self.step_net[-1].bias.zero_()

    def accel_next(self, s_prev, s_next, u_next, h, ctx=None):
        """B2 has no ṡ: v̇ ≈ Δv/Δt of its own prediction (documented asymmetry with B3)."""
        return (s_next[..., 3:6] - s_prev[..., 3:6]) / self.dt

    def transition(self, s, u, h, ctx=None):
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
        feat = dec_layers(cfg)[-1] if (cfg.shared_trunk and dec_layers(cfg)) else cfg.latent        # size of what d_θ reads of h
        self.resid = _mlp(feat + 10 + (4 if cfg.resid_uses_u else 0) + self.ORACLE_DIM, [w] * cfg.resid_depth, 12, cfg.activation)
        with torch.no_grad():
            self.resid[-1].weight.mul_(0.05); self.resid[-1].bias.zero_()
        self.register_buffer("res_scale", _residual_scale(stats, dt))
        self.register_buffer("res_mask", _residual_mask(cfg.residual_channels))

    ORACLE_DIM = 0       # extra residual inputs (B6b: the recorded wrench, 6)

    def _feat(self, h: torch.Tensor) -> torch.Tensor:
        """What the residual reads of the latent: h itself, or the decoder's trunk T(h) when ``shared_trunk`` (option c)."""
        return self.dec[:-1](h) if self.cfg.shared_trunk else h

    def _resid_in(self, h, s, u, orc=None) -> torch.Tensor:
        parts = [self._feat(h), self._state_in(s)]
        if self.cfg.resid_uses_u:
            parts.append(self._u(u))
        if orc is not None:
            parts.append(orc)
        return torch.cat(parts, -1)

    def residual(self, h: torch.Tensor, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.resid(self._resid_in(h, s, u)) * self.res_scale * self.res_mask

    def residual_sq(self, h, s, u, ctx=None):
        r = self.resid(self._resid_in(h, s, u)) * self.res_mask
        return r.square().sum(-1) / self.res_mask.sum()

    def residual_zero_sq(self, h, s, u, ctx=None):
        r = self.resid(self._resid_in(torch.zeros_like(h), s, u)) * self.res_mask
        return r.square().sum(-1) / self.res_mask.sum()

    def accel_next(self, s_prev, s_next, u_next, h, ctx=None):
        """v̇ at t+1 from the model's own ṡ = f_phys + d_θ, evaluated at (s̃_{t+1}, u_{t+1}, h_t) — the stored IMU is
        instantaneous (obs_mode "instant")."""
        return self.physics.derivative(s_next, u_next, self.residual(h, s_next, u_next))[..., 3:6]

    def transition(self, s, u, h, ctx=None):
        return rk4_step(lambda x: self.physics.derivative(x, u, self.residual(h, x, u)), s, self.dt, self.physics.project)


class PureResidualLatent(_LatentBase):
    """B5: no physics knowledge in the dynamics; same encoder, latent, decoder and φ coupling as B3, integrated with RK4.

    * variant ``"a"`` (``b5a``): fully learned vector field, ``ṡ = d_θ(h, s, u)`` over all 12 channels
      (ṗ, v̇, attitude rate δ, ω̇): ``q̇ = ½ q ⊗ [0, δ]``.
    * variant ``"b"`` (``b5b``, recommended in ``.idea/baselines``): exact kinematics ``ṗ = v``, ``q̇ = ½ q ⊗ ω_B`` and a learned
      full ``v̇``, ``ω̇`` (gravity, thrust mapping and gyroscopic terms are learned, not given).

    Output scale = std of the per-step change of the *full* state / Δt (the same statistics B3 uses for its residual): for B5 these
    are the magnitudes of the full derivative, so the network output is O(1) in standardised units. Width is auto-matched to B3's
    total parameters (``ModelConfig.b5_hidden`` overrides, e.g. to run a wider B5). No APHYNITY penalty (nothing to keep small)."""

    def __init__(self, stats, variant: str, cfg: ModelConfig | None = None, dt: float = 0.02) -> None:
        if variant not in ("a", "b"):
            raise ValueError("variant must be 'a' or 'b'")
        cfg = cfg or ModelConfig()
        super().__init__(stats, cfg, dt)
        self.variant = variant
        self.name = "b5" + variant
        w = cfg.b5_hidden or _auto_width("b5", stats, cfg, dt)
        self.field = _mlp(cfg.latent + 10 + 4, [w] * cfg.field_depth, 12, cfg.activation)
        with torch.no_grad():
            self.field[-1].weight.mul_(0.05); self.field[-1].bias.zero_()
        self.register_buffer("res_scale", _residual_scale(stats, dt))
        mask = torch.ones(12) if variant == "a" else _residual_mask("acc_only")
        self.register_buffer("res_mask", mask)

    @staticmethod
    def _project(s: torch.Tensor) -> torch.Tensor:
        return torch.cat([s[..., 0:6], quat.normalize(s[..., 6:10]), s[..., 10:13]], -1)

    def field_out(self, h, s, u):
        return self.field(torch.cat([h, self._state_in(s), self._u(u)], -1)) * self.res_scale * self.res_mask

    def deriv(self, h, s, u):
        d = self.field_out(h, s, u)
        q = quat.normalize(s[..., 6:10])
        if self.variant == "a":
            pdot, w_kin = d[..., 0:3], d[..., 6:9]
        else:
            pdot, w_kin = s[..., 3:6], s[..., 10:13]
        return torch.cat([pdot, d[..., 3:6], quat.omega_matrix_derivative(q, w_kin), d[..., 9:12]], -1)

    def accel_next(self, s_prev, s_next, u_next, h, ctx=None):
        return self.field_out(h, s_next, u_next)[..., 3:6]

    def transition(self, s, u, h, ctx=None):
        return rk4_step(lambda x: self.deriv(h, x, u), s, self.dt, self._project)


ORACLE_FORCE_SCALE = 9.81      # F/(m g): dimensionless input of B6b (proposed fixed scales; no data statistics needed)
ORACLE_ANGACC_SCALE = 10.0     # J⁻¹τ [rad/s²] / 10


def _oracle_at(batch: Mapping[str, torch.Tensor], idx: int, physics: QuadrotorPhysics, timing: str):
    """(force_W/m, J⁻¹τ_B) of the recorded total wrench for step ``idx`` ("future") or step 0 held ("current")."""
    i = idx if timing == "future" else 0
    f = batch["oracle_force_W"][:, i] / physics.mass
    a = batch["oracle_torque_B"][:, i] @ physics.J_inv.T
    return f, a


class PhysicsOracle(nn.Module):
    """B6a: ``ṡ = f_phys(ŝ, u) + d_oracle``, ``d_oracle = [0, F_ext/m_nom, 0, J_nom⁻¹ τ_ext]`` from the recorded wrench. Nothing is learned.

    ``timing="future"``: the true wrench of every step; ``"current"``: the wrench of step 0 held over the horizon (a perfect ``h`` that
    never updates). The nominal mass and inertia are used, so parameter mismatch stays uncorrected (upper bound for the aero part only)."""
    needs_oracle = True

    def __init__(self, physics: QuadrotorPhysics, timing: str = "future", dt: float = 0.02) -> None:
        super().__init__()
        if timing not in ("future", "current"):
            raise ValueError("timing must be 'future' or 'current'")
        self.physics, self.timing, self.dt = physics, timing, float(dt)
        self.name = f"b6a_{timing}"

    @torch.no_grad()
    def predict(self, batch, horizon: int, initial_state: str = "given", anchor: Anchor | None = None) -> torch.Tensor:
        if initial_state != "given":
            raise ValueError("B6 has no encoder/decoder: initial_state must be 'given'")
        s, outs = self.physics.project(batch["state_history"][:, -1]), []
        for k in range(horizon):
            u = batch["future_actions"][:, k]
            f, a = _oracle_at(batch, k, self.physics, self.timing)
            extra = torch.zeros(s.shape[0], 12, device=s.device, dtype=s.dtype)
            extra[:, 3:6], extra[:, 9:12] = f, a
            s = rk4_step(lambda x: self.physics.derivative(x, u, extra), s, self.dt, self.physics.project)
            outs.append(s)
            if anchor is not None and anchor.state and anchor.fires(k, horizon):
                s = batch["future_states"][:, k]
        return torch.stack(outs, dim=1)

    def forward(self, batch, horizon: int, initial_state: str = "given", anchor: Anchor | None = None) -> dict:
        return dict(pred=self.predict(batch, horizon, initial_state, anchor), dec=None)


class PhysicsResidualOracle(PhysicsResidualLatent):
    """B6b: B3 with the recorded wrench as an extra input of ``d_θ`` (same network otherwise; not parameter-matched).
    ``timing`` as in B6a. The oracle also enters φ through ``accel_next`` (v̇ includes the residual), like the real IMU would."""
    needs_oracle = True
    ORACLE_DIM = 6

    def __init__(self, stats, timing: str = "future", cfg: ModelConfig | None = None, dt: float = 0.02,
                 physics: QuadrotorPhysics | None = None) -> None:
        if timing not in ("future", "current"):
            raise ValueError("timing must be 'future' or 'current'")
        super().__init__(stats, cfg, dt, physics)
        self.timing = timing
        self.name = f"b6b_{timing}"

    def _orc(self, ctx, offset: int) -> torch.Tensor:
        batch, k = ctx
        f, a = _oracle_at(batch, k + offset, self.physics, self.timing)
        return torch.cat([f / ORACLE_FORCE_SCALE, a / ORACLE_ANGACC_SCALE], -1)      # (F/m)/g and J⁻¹τ/10

    def residual(self, h, s, u, orc=None):
        if orc is None:
            raise ValueError("B6b: the oracle wrench input `orc` must be provided (use `self._orc(ctx, offset)`)")
        return self.resid(self._resid_in(h, s, u, orc)) * self.res_scale * self.res_mask

    def residual_sq(self, h, s, u, ctx=None):
        r = self.resid(self._resid_in(h, s, u, self._orc(ctx, 0))) * self.res_mask
        return r.square().sum(-1) / self.res_mask.sum()

    def residual_zero_sq(self, h, s, u, ctx=None):
        r = self.resid(self._resid_in(torch.zeros_like(h), s, u, self._orc(ctx, 0))) * self.res_mask
        return r.square().sum(-1) / self.res_mask.sum()

    def accel_next(self, s_prev, s_next, u_next, h, ctx=None):
        extra = self.residual(h, s_next, u_next, self._orc(ctx, 1))
        return self.physics.derivative(s_next, u_next, extra)[..., 3:6]

    def transition(self, s, u, h, ctx=None):
        orc = self._orc(ctx, 0)
        return rk4_step(lambda x: self.physics.derivative(x, u, self.residual(h, x, u, orc)), s, self.dt, self.physics.project)


class PhysicsResidualNoLatent(nn.Module):
    """B4: physics + residual **without** latent memory: ``d_θ(s_t, u_t)``. No encoder, decoder or latent, so the
    start state is the given one. Width is auto-matched to B3's parameter count (thesis: isolate the memory)."""
    name = "b4"

    def __init__(self, stats, cfg: ModelConfig | None = None, dt: float = 0.02, physics: QuadrotorPhysics | None = None,
                 width: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig(); self.dt = float(dt)
        for k in STAT_KEYS:
            self.register_buffer(k, torch.tensor(stats[k], dtype=torch.float32))
        self.physics = physics or QuadrotorPhysics.from_yaml()
        w = width or self.cfg.b4_hidden or _auto_width("b4", stats, self.cfg, dt)
        self.resid = _mlp(10 + (4 if self.cfg.resid_uses_u else 0), [w] * self.cfg.b4_depth, 12, self.cfg.activation)
        with torch.no_grad():
            self.resid[-1].weight.mul_(0.05); self.resid[-1].bias.zero_()
        self.register_buffer("res_scale", _residual_scale(stats, dt))
        self.register_buffer("res_mask", _residual_mask(self.cfg.residual_channels))

    def _u(self, u): return (u - self.act_mu) / self.act_sd

    def _state_in(self, s): return torch.cat([(s[..., 3:6] - self.v_mu) / self.v_sd, s[..., 6:10], (s[..., 10:13] - self.w_mu) / self.w_sd], -1)

    def _resid_in(self, s, u):
        return torch.cat([self._state_in(s), self._u(u)], -1) if self.cfg.resid_uses_u else self._state_in(s)

    def residual(self, s, u):
        return self.resid(self._resid_in(s, u)) * self.res_scale * self.res_mask

    def residual_sq(self, s, u):
        r = self.resid(self._resid_in(s, u)) * self.res_mask
        return r.square().sum(-1) / self.res_mask.sum()

    def forward(self, batch, horizon: int, initial_state: str = "given", anchor: Anchor | None = None) -> dict:
        if initial_state != "given":
            raise ValueError("B4 has no encoder/decoder: initial_state must be 'given'")
        p_ref, _ = relative_frame(batch)
        s = self.physics.project(_shift(batch["state_history"][:, -1], p_ref[:, 0], -1.0))
        outs, rsq = [], []
        for k in range(horizon):
            u = batch["future_actions"][:, k]
            rsq.append(self.residual_sq(s, u))
            s = rk4_step(lambda x: self.physics.derivative(x, u, self.residual(x, u)), s, self.dt, self.physics.project)
            outs.append(s)
            if anchor is not None and anchor.state and anchor.fires(k, horizon):
                s = _shift(batch["future_states"][:, k], p_ref[:, 0], -1.0)
        return dict(pred=_shift(torch.stack(outs, 1), p_ref, 1.0), dec=None, resid_sq=torch.stack(rsq, 1))

    def predict(self, batch, horizon: int, initial_state: str = "given", anchor: Anchor | None = None) -> torch.Tensor:
        return self(batch, horizon, initial_state, anchor)["pred"]


def set_phi_noise(model: nn.Module, preset: str) -> None:
    """Match the white noise added to φ's synthetic IMU to the noise of the real IMU being fed (no-op for B0/B1/B4)."""
    if hasattr(model, "phi_noise"):
        model.phi_noise = preset


def n_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _auto_width(kind: str, stats, cfg: ModelConfig, dt: float) -> int:
    """Hidden width (``*_depth`` hidden layers) for B2's transition / B4's residual so that the parameter count is as
    close as possible to B3's (thesis: 'approximately comparable parameter capacity')."""
    target = n_parameters(PhysicsResidualLatent(stats, cfg, dt))
    base = n_parameters(_LatentBase(stats, cfg, dt)) if kind in ("b2", "b5") else 0
    best, best_gap = 16, float("inf")
    for w in range(8, 1025, 2):
        if kind in ("b2", "b5"):
            depth = cfg.trans_depth if kind == "b2" else cfg.field_depth
            n = base + mlp_params(cfg.latent + 14, [w] * depth, 12)
        else:   # b4
            n = mlp_params(10 + (4 if cfg.resid_uses_u else 0), [w] * cfg.b4_depth, 12)
        if abs(n - target) < best_gap:
            best, best_gap = w, abs(n - target)
    if best_gap > 0.10 * target:
        warnings.warn(f"{kind}: closest width {best} is {100 * best_gap / target:.1f}% away from "
                      f"B3's parameter count ({target}); outside the ±10% matching tolerance", stacklevel=2)
    return best


MATCH_TOL = 0.10


def parameter_table(stats, cfg: ModelConfig, dt: float, physics: QuadrotorPhysics | None = None) -> dict:
    """Parameter counts of every latent / residual model for one configuration, with the gap to B3.

    b4 = width matched to B3's total; b4_rw = B3's residual width (``cfg.resid_hidden``); b2 = matched to B3's total."""
    kinds = {k: build_model(k, stats=stats, physics=physics, dt=dt, cfg=cfg) for k in ("b2", "b3", "b4", "b4_rw", "b5a", "b5b")}
    n = {k: n_parameters(m) for k, m in kinds.items()}
    return {k: {"params": v, "vs_b3_pct": 100.0 * (v - n["b3"]) / n["b3"],
                "matched": k in ("b2", "b4", "b5a", "b5b"), "within_tol": abs(v - n["b3"]) <= MATCH_TOL * n["b3"]} for k, v in n.items()}


def check_obs_mode(model: nn.Module, obs_mode: str) -> None:
    """φ synthesises the *instantaneous* IMU, so models that use it must be fed ``obs_mode="instant"`` data."""
    if hasattr(model, "synth_imu") and obs_mode != "instant":
        raise ValueError(f"{getattr(model, 'name', type(model).__name__)} uses φ (instantaneous synthetic IMU) but the data "
                         f"were built with obs_mode={obs_mode!r}; use obs_mode='instant'")


def build_model(kind: str, *, stats=None, physics: QuadrotorPhysics | None = None, dt: float = 0.02,
                cfg: ModelConfig | None = None) -> nn.Module:
    kind = kind.lower()
    if kind == "b0":
        return HoldLastState()
    if kind == "b1":
        return PhysicsOnly(physics or QuadrotorPhysics.from_yaml(), dt)
    if kind in ("b5a", "b5b"):
        if stats is None:
            raise ValueError(f"{kind} needs normalisation statistics (fit_stats on the training episodes)")
        return PureResidualLatent(stats, kind[-1], cfg, dt)
    if kind in ("b6a_future", "b6a_current"):
        return PhysicsOracle(physics or QuadrotorPhysics.from_yaml(), kind.split("_")[1], dt)
    if kind in ("b6b_future", "b6b_current"):
        if stats is None:
            raise ValueError(f"{kind} needs normalisation statistics (fit_stats on the training episodes)")
        return PhysicsResidualOracle(stats, kind.split("_")[1], cfg, dt, physics)
    if kind in ("b2", "b3", "b4", "b4_rw"):
        if stats is None:
            raise ValueError(f"{kind} needs normalisation statistics (fit_stats on the training episodes)")
        if kind == "b4_rw":     # B4 with B3's residual width (not parameter-matched; the other B4 is matched to B3's total)
            c = cfg or ModelConfig()
            return PhysicsResidualNoLatent(stats, c, dt, physics, width=c.resid_hidden)
        cls = {"b2": PureNeuralLatent, "b3": PhysicsResidualLatent, "b4": PhysicsResidualNoLatent}[kind]
        return cls(stats, cfg, dt) if kind == "b2" else cls(stats, cfg, dt, physics)
    raise ValueError(f"unknown model {kind!r}; implemented: b0, b1, b2, b3, b4, b4_rw, b5a, b5b, b6a_future, b6a_current, b6b_future, b6b_current")
