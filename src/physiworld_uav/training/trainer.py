"""Training loop for B2 (and later B3–B5) on the model-ready pilot data.

Features (docs/specs/training_pipeline.md): AdamW with warm-up + cosine schedule, gradient clipping,
horizon curriculum (thesis Stage 5), validation on ``val_id`` each epoch, early stopping, atomic
``last.pt`` / ``best.pt`` / ``epoch_*.pt`` checkpoints, exact resume, ``metrics.jsonl``, ``provenance.json``.
A crash loses at most one epoch.
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from physiworld_uav.data.dataset import ModelReadyDataset
from physiworld_uav.data.noise import ImuNoise
from physiworld_uav.data.dataset_builder import (IMU_RAW_WARNING, BuilderConfig, dataset_decimation, imu_transport_status,
                                                 load_model_episodes)
from physiworld_uav.evaluation.evaluator import evaluate
from physiworld_uav.losses.losses import LossWeights, fit_loss_weights, total_loss, weight_table
from physiworld_uav.models.world_model import (STAT_KEYS, ModelConfig, build_model, check_obs_mode, fit_stats,
                                               parameter_table, set_phi_noise)
from . import checkpoint as ckpt
from .resources import ResourceMeter
from .tracking import Tracker

BATCH_KEYS = ("state_history", "obs_history", "action_history", "future_actions", "future_states")


@dataclass
class TrainConfig:
    kind: str = "b2"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    warmup_steps: int = 200
    grad_clip: float = 1.0
    patience: int = 10                    # early stopping on the validation metric (last stage only)
    stage_patience: int = 3               # plateau epochs before advancing the horizon curriculum
    stage_max_epochs: int = 0             # 0 = no cap
    horizon_stages: list = field(default_factory=lambda: [1, 5, 10, 20, 50])   # thesis curriculum
    val_horizon: int = 50                 # validation / early-stopping horizon (capped at the curriculum maximum)
    initial_state: str = "given"
    gamma: float = 0.99
    lambda_roll: float = 1.0
    lambda_dec: float = 1.0               # decoder reconstruction weight (1.0 = previous behaviour)
    lambda_resid: float = 1e-3            # APHYNITY-style ‖d_θ‖² penalty (0 = off); an ablation knob, not tuned
    lambda_zero: float = 0.0              # λ₀ ‖d_θ(h=0)‖² soft zero-residual penalty (0 = off; spec §3.4 option b, proposed 1e-3)
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"
    tiny_episodes: int = 0                # >0: overfit test on the first N training episodes, no validation
    # ---- model sizes: one-to-one with ``models.world_model.ModelConfig`` (a test keeps names and defaults identical)
    latent: int = 128
    embed: int = 64
    hidden: int = 256
    enc_hidden: list | None = None        # encoder hidden layers; None = [hidden // 2]
    dec_hidden: list | None = None        # decoder hidden layers; None = [hidden]
    trans_hidden: int = 0                 # B2 transition width; 0 = auto-match B3's parameter count
    resid_hidden: int = 256               # B3 / B6b residual width (and B4 (B3 width))
    b4_hidden: int = 0                    # B4 residual width; 0 = auto-match B3's parameter count
    b5_hidden: int = 0                    # B5 field width; 0 = auto-match B3's parameter count
    trans_depth: int = 2
    resid_depth: int = 2
    b4_depth: int = 2
    field_depth: int = 2
    activation: str = "silu"
    resid_uses_u: bool = True             # False = residual without the action (spec §3.4 option a)
    shared_trunk: bool = False            # True = residual reads the decoder trunk (spec §3.4 option c)
    residual_channels: str = "acc_only"   # "all" = unrestricted ablation
    known_vehicle: bool = False           # True = f_phys uses each episode's TRUE mass, k_F, k_M, inertia (B1, B3, B4, B6); needs PhysX read-back in the data
    vehicle_input: bool = False           # True = B2 / B5 also get the vehicle parameters as input of their dynamics network
    enc_uses_u: bool = False              # True restores u_t as an input of E (ablation, B2 and B3)
    obs_mode: str = "instant"             # φ needs "instant"
    history: int = 10
    keep_epoch_checkpoints: int = 3
    train_extra_statuses: list = field(default_factory=list)   # e.g. ["terminated_ground", "terminated_envelope"]: early-ended episodes used for TRAINING only
    tail_trim_ground: int = 5             # model steps dropped from the end of a ground-contact episode (the impact)
    fit_on: str = "all"                   # "closed_loop": normalisation statistics and loss weights are fitted on the closed-loop training episodes only
    noise: str = "nominal"                 # IMU noise preset for training windows (data/noise.py); "none" = sanity run only
    val_noise: str = "nominal"
    use_tensorboard: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def model_cfg_of(cfg: TrainConfig) -> ModelConfig:
    """Every ModelConfig field that TrainConfig also has is copied by name (so a new size knob cannot be forgotten here)."""
    shared = {f.name: getattr(cfg, f.name) for f in fields(ModelConfig) if f.name != "phi_noise" and hasattr(cfg, f.name)}
    return ModelConfig(**shared, phi_noise=cfg.noise)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def dataset_identity(dataset_dir: Path) -> dict:
    ident: dict = {"dir": str(dataset_dir)}
    for name, keys in (("index.json", ("dataset_id", "dataset_version")), ("provenance.json", ("config_hash",))):
        p = Path(dataset_dir) / name
        if p.exists():
            d = json.loads(p.read_text())
            ident.update({k: d.get(k) for k in keys})
    return ident


def _batch(b: dict, device) -> dict:
    from physiworld_uav.data.dataset import ORACLE_KEYS
    return {k: b[k].to(device) for k in BATCH_KEYS + tuple(k for k in ORACLE_KEYS + ("vehicle",) if k in b)}


def make_loader(ds, cfg: TrainConfig, epoch: int) -> DataLoader:
    g = torch.Generator().manual_seed(cfg.seed * 100003 + epoch)      # per-epoch order: resume is exact
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=len(ds) >= cfg.batch_size,
                      num_workers=cfg.num_workers, generator=g, pin_memory=torch.cuda.is_available())


def lr_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def build_from_checkpoint(state: dict) -> torch.nn.Module:
    m = build_model(state["kind"], stats=state["stats"], cfg=ModelConfig(**state["model_cfg"]), dt=state["dt"])
    m.load_state_dict(state["model"])
    return m


def train(dataset_dir: str | Path, out_dir: str | Path, cfg: TrainConfig, *, resume: bool = False,
          force: bool = False, log=print, max_epochs: int | None = None,
          episodes: tuple | None = None) -> dict:
    """Train; returns a summary dict. ``max_epochs`` stops early (used by tests to simulate a crash)."""
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = pick_device(cfg.device)
    if cfg.kind.startswith("b6a"):
        raise ValueError("b6a has no parameters: evaluate it with `evaluation.run --models b6a_future` instead of training")
    needs_oracle = cfg.kind.startswith("b6")        # B6b only: the recorded wrench is an input (reference upper bound)
    P = int(max(cfg.horizon_stages))
    dec = dataset_decimation(dataset_dir)          # 4 for the 50 Hz pilots, 10 for a 20 Hz dataset (read from the manifest)
    extra = tuple(cfg.train_extra_statuses)
    tail = (("terminated_ground", int(cfg.tail_trim_ground)),) if cfg.tail_trim_ground > 0 and "terminated_ground" in extra else ()

    def make_bcfg(h: int) -> BuilderConfig:        # early-ended episodes are eligible in the training role only (BuilderConfig.extra_status_roles)
        return BuilderConfig(decimation=dec, history=cfg.history, horizon=int(h), obs_mode=cfg.obs_mode, extra_statuses=extra, tail_trim=tail)

    bcfg = make_bcfg(P)

    if episodes is not None:                       # (train_eps, val_eps) injected, e.g. by tests
        train_eps, val_eps = list(episodes[0]), list(episodes[1])
    else:
        train_eps = load_model_episodes(dataset_dir, bcfg, roles=["train"])
        val_eps = load_model_episodes(dataset_dir, bcfg, roles=["val_id"])
    if cfg.tiny_episodes:
        train_eps = sorted(train_eps, key=lambda e: e.episode_id)[: cfg.tiny_episodes]
        val_eps = []

    def stage_cfg(h: int) -> BuilderConfig:        # windows are built per curriculum stage: stage h needs only h future steps
        return make_bcfg(h)

    train_noise = ImuNoise(cfg.noise, cfg.seed, deterministic=False) if cfg.noise != "none" else None
    _stage_ds: dict = {}

    def train_ds_for(h: int) -> ModelReadyDataset:
        if h not in _stage_ds:
            ds = ModelReadyDataset(train_eps, stage_cfg(h), transform=train_noise, oracle=needs_oracle)
            if len(ds) == 0:
                raise ValueError(f"no training windows for horizon {h} (episodes too short for history {cfg.history} + horizon {h})")
            _stage_ds[h] = ds
        return _stage_ds[h]

    val_k = int(cfg.val_horizon)                    # one fixed validation set (same windows for every stage and every model)
    val_ds = ModelReadyDataset(val_eps, stage_cfg(val_k), transform=ImuNoise(cfg.val_noise, 12345) if cfg.val_noise != "none" else None,
                               oracle=needs_oracle)
    train_ds = train_ds_for(int(cfg.horizon_stages[0]))
    log(f"train: {len(train_eps)} episodes, windows per stage "
        + ", ".join(f"H={h}: {len(train_ds_for(int(h)))}" for h in cfg.horizon_stages)
        + f"; val: {len(val_eps)} episodes, {len(val_ds)} windows at H={val_k}; device {device}")
    ident = dataset_identity(dataset_dir)

    state = None
    last = out_dir / "checkpoints" / "last.pt"
    if resume and last.exists():
        state = ckpt.load(last)
        if state["dataset"] != ident and not force:
            raise RuntimeError(f"checkpoint was trained on {state['dataset']}, now {ident}; pass force=True to override")
        if {**asdict(ModelConfig()), **state["model_cfg"]} != asdict(model_cfg_of(cfg)) and not force:   # older checkpoints lack newer size keys
            raise RuntimeError("model configuration differs from the checkpoint")
        model = build_from_checkpoint(state); weights = LossWeights(**state["loss_weights"]); winfo = None
        log(f"resuming from epoch {state['epoch']} (step {state['step']})")
    else:
        if cfg.fit_on not in ("all", "closed_loop"):
            raise ValueError("fit_on must be 'all' or 'closed_loop'")
        fit_eps = [e for e in train_eps if e.meta is None or not (e.meta.plan or {}).get("open_loop")] if cfg.fit_on == "closed_loop" else train_eps
        if not fit_eps:
            raise ValueError("fit_on='closed_loop' but there is no closed-loop training episode")
        by_status = {}
        for e in train_eps:
            st = getattr(e.meta, "status", "?") if e.meta is not None else "?"
            by_status[st] = by_status.get(st, 0) + 1
        log(f"training episodes by status: {by_status}; statistics and loss weights fitted on {len(fit_eps)} episodes ({cfg.fit_on})")
        stats = fit_stats(fit_eps)
        dt0 = train_eps[0].dt
        model = build_model(cfg.kind, stats=stats, cfg=model_cfg_of(cfg), dt=dt0)
        hw = min(20, P)                              # weights are fitted on hold-last-state errors at this horizon
        weights, winfo = fit_loss_weights(ModelReadyDataset(fit_eps, stage_cfg(hw)), hw, gamma=cfg.gamma, lambda_roll=cfg.lambda_roll,
                                          seed=cfg.seed, resid=cfg.lambda_resid, dec=cfg.lambda_dec, zero=cfg.lambda_zero, return_info=True)
        ptable = parameter_table(stats, model_cfg_of(cfg), dt0)
        log("parameter counts (B2/B4 matched to B3's total within ±10%; b4_rw = B3's residual width):\n"
            + "\n".join(f"  {k:6s} {v['params']:>9d}  ({v['vs_b3_pct']:+6.1f}% vs b3){'  matched' if v['matched'] else ''}"
                        f"{'' if v['within_tol'] or not v['matched'] else '  OUTSIDE ±10%'}" for k, v in ptable.items()))
        (out_dir / "param_counts.json").write_text(json.dumps(ptable, indent=2))
    check_obs_mode(model, cfg.obs_mode)
    if hasattr(model, "synth_imu") and imu_transport_status(train_eps) in ("raw", "mixed"):
        log("WARNING: " + IMU_RAW_WARNING)
    model.to(device)
    model.collect_zero_resid = bool(weights.zero)        # option b: the model evaluates d_θ(h = 0) only when it is penalised
    log(f"loss weights: {weights.to_dict()}")
    if winfo:
        log("per-component loss weights (before training):\n" + weight_table(winfo))
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_ds) // cfg.batch_size)      # first (largest) stage; the cosine schedule is approximate
    total_steps = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total_steps, cfg.warmup_steps))

    epoch, step, stage, bad, stage_epochs, best = 0, 0, 0, 0, 0, float("inf")
    # ``best`` is the best metric of the CURRENT curriculum stage (it drives patience and stage advance and is reset at each
    # stage change). ``best_global`` / ``best_epoch`` is the best validation metric of the whole run; it alone decides best.pt.
    best_global, best_epoch = float("inf"), 0
    if state is not None:
        opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        epoch, step, stage, bad, stage_epochs, best = (state[k] for k in ("epoch", "step", "stage", "bad", "stage_epochs", "best"))
        best_global, best_epoch = state.get("best_global", float("inf")), state.get("best_epoch", 0)   # older checkpoints lack them
        ckpt.set_rng_state(state["rng"])
    else:
        (out_dir / "config.yaml").write_text(json.dumps({"train": asdict(cfg), "builder": asdict(bcfg)}, indent=2))
        (out_dir / "provenance.json").write_text(json.dumps(
            {"git": _git_commit(), "torch": torch.__version__, "dataset": ident, "n_params": n_params,
             "started": time.strftime("%Y-%m-%d %H:%M:%S"), "loss_weights": weights.to_dict(),
             "loss_weight_info": winfo, "param_counts": ptable}, indent=2))

    tracker = Tracker(out_dir, out_dir.name, {"model": cfg.kind, "dataset": f"{ident.get('dataset_id')}_v{ident.get('dataset_version')}",
                                              "n_params": n_params, "git": _git_commit(), "noise": cfg.noise, "resumed": state is not None},
                      use_tensorboard=cfg.use_tensorboard)
    metrics_path = out_dir / "metrics.jsonl"
    meter = ResourceMeter(device)
    t_run0 = time.time()
    done = False
    stop_reason = "max_epochs"
    while epoch < cfg.epochs and not done:
        h = int(cfg.horizon_stages[stage])
        model.train(); set_phi_noise(model, cfg.noise)
        meter.start()
        t0, agg, nb = time.time(), {}, 0
        for batch in make_loader(train_ds_for(h), cfg, epoch):
            b = _batch(batch, device)
            out = model(b, h, cfg.initial_state)
            parts = total_loss(out, b, h, weights)
            opt.zero_grad(set_to_none=True)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step(); sched.step(); step += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            nb += 1
        epoch += 1; stage_epochs += 1
        rec = {"epoch": epoch, "step": step, "horizon": h, "stage": stage, "lr": sched.get_last_lr()[0],
               "train": {k: v / max(nb, 1) for k, v in agg.items()}, "epoch_s": time.time() - t0,
               "windows_per_s": nb * cfg.batch_size / max(time.time() - t0, 1e-9)}
        t_val = time.time()
        if len(val_ds):
            set_phi_noise(model, cfg.val_noise)
            ev = evaluate(model, val_ds, horizon=val_k, device=device, batch_size=512, initial_state=cfg.initial_state, n_boot=0,
                          ks=sorted({1, 5, 10, 20, val_k}))
            rec["val"] = {"pos_rmse_m": ev["pos_rmse_m"], "pos_mean_m": ev["pos_mean_m"], "att_rmse_deg": ev["att_rmse_deg"],
                          "reliable_horizon": ev["reliable_horizon_mean"]}
            metric = ev["pos_rmse_m"][str(val_k)]
        else:
            metric = rec["train"]["total"]
        improved = metric < best - 1e-12
        if improved:
            best, bad = metric, 0
        else:
            bad += 1
        # best.pt: the lowest validation metric over ALL epochs (the validation horizon is the same in every stage, so epochs of
        # different stages are comparable). Without a validation set the metric is the training loss of the stage horizon, which
        # is not comparable across stages: then best.pt follows the per-stage ``improved``.
        improved_global = (metric < best_global - 1e-12) if len(val_ds) else improved
        if improved_global:
            best_global, best_epoch = metric, epoch
        # timing for live monitoring: epoch_s / windows_per_s are TRAINING only; val_s is validation; elapsed_s since the run began
        rec["val_s"] = time.time() - t_val
        rec["n_train_episodes"] = len(train_eps)
        rec["n_train_windows"] = nb * cfg.batch_size            # windows seen in this epoch (drop_last): basis for scaling to a larger dataset
        rec["s_per_episode"] = rec["epoch_s"] / max(len(train_eps), 1)          # training seconds per training episode (one pass)
        rec["episodes_per_s"] = len(train_eps) / max(rec["epoch_s"], 1e-9)
        rec["elapsed_s"] = time.time() - t_run0
        rec["resources"] = meter.stop()
        rec["metric"], rec["best"], rec["best_global"] = metric, best, best_global
        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        tracker.epoch(rec)
        log(f"epoch {epoch:3d} h={h:2d} loss {rec['train']['total']:.4g} metric {metric:.4g} best {best:.4g} ({rec['epoch_s']:.1f}s)")

        last_stage = stage >= len(cfg.horizon_stages) - 1
        if not last_stage and (bad >= cfg.stage_patience or (cfg.stage_max_epochs and stage_epochs >= cfg.stage_max_epochs)):
            stage, bad, stage_epochs, best = stage + 1, 0, 0, float("inf")      # best is horizon-specific: restart
            log(f"curriculum: advancing to horizon {cfg.horizon_stages[stage]}")
        elif last_stage and bad >= cfg.patience:
            done, stop_reason = True, "early_stopping"

        snap = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "epoch": epoch, "step": step, "stage": stage, "bad": bad, "stage_epochs": stage_epochs, "best": best,
                "best_global": best_global, "best_epoch": best_epoch,
                "rng": ckpt.rng_state(), "stats": {k: getattr(model, k).tolist() for k in STAT_KEYS},
                "kind": cfg.kind, "model_cfg": asdict(model_cfg_of(cfg)), "dt": model.dt,
                "loss_weights": weights.to_dict(), "dataset": ident, "history": cfg.history, "obs_mode": cfg.obs_mode, "train_cfg": asdict(cfg), "git": _git_commit(),
                "metric": metric}
        ckpt.save_atomic(snap, last)
        ckpt.save_atomic(snap, out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")
        if improved_global:
            ckpt.save_atomic(snap, out_dir / "checkpoints" / "best.pt")
        ckpt.prune(out_dir / "checkpoints", cfg.keep_epoch_checkpoints)
        if max_epochs is not None and epoch >= max_epochs:
            break
    else:
        stop_reason = "epochs_done" if epoch >= cfg.epochs else stop_reason
    summary = {"epochs": epoch, "best_metric": best_global, "best_epoch": best_epoch, "stop_reason": stop_reason, "n_params": n_params, "out_dir": str(out_dir),
               "total_wall_s": time.time() - t_run0, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
               "peak_gpu_reserved_mb": meter.peak_gpu_mb, "peak_rss_mb": meter.peak_rss_mb, "device": str(device)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tracker.end({**summary, "model": cfg.kind})
    return summary
