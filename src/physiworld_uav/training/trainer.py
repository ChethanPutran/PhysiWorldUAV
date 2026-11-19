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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from physiworld_uav.data.dataset import ModelReadyDataset
from physiworld_uav.data.noise import ImuNoise
from physiworld_uav.data.dataset_builder import BuilderConfig, load_model_episodes
from physiworld_uav.evaluation.evaluator import evaluate
from physiworld_uav.losses.losses import LossWeights, fit_loss_weights, total_loss
from physiworld_uav.models.world_model import STAT_KEYS, ModelConfig, build_model, fit_stats
from . import checkpoint as ckpt
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
    horizon_stages: list = field(default_factory=lambda: [20])
    initial_state: str = "given"
    gamma: float = 0.99
    lambda_roll: float = 1.0
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"
    tiny_episodes: int = 0                # >0: overfit test on the first N training episodes, no validation
    latent: int = 128
    embed: int = 64
    hidden: int = 256
    residual_channels: str = "all"
    history: int = 10
    keep_epoch_checkpoints: int = 3
    noise: str = "none"                    # IMU noise preset applied to training windows (data/noise.py)
    val_noise: str = "none"
    use_tensorboard: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def model_cfg_of(cfg: TrainConfig) -> ModelConfig:
    return ModelConfig(latent=cfg.latent, embed=cfg.embed, hidden=cfg.hidden, residual_channels=cfg.residual_channels)


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
    return {k: b[k].to(device) for k in BATCH_KEYS}


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
    P = int(max(cfg.horizon_stages))
    bcfg = BuilderConfig(decimation=4, history=cfg.history, horizon=P)

    if episodes is not None:                       # (train_eps, val_eps) injected, e.g. by tests
        train_eps, val_eps = list(episodes[0]), list(episodes[1])
    else:
        train_eps = load_model_episodes(dataset_dir, bcfg, roles=["train"])
        val_eps = load_model_episodes(dataset_dir, bcfg, roles=["val_id"])
    if cfg.tiny_episodes:
        train_eps = sorted(train_eps, key=lambda e: e.episode_id)[: cfg.tiny_episodes]
        val_eps = []
    train_ds = ModelReadyDataset(train_eps, bcfg, transform=ImuNoise(cfg.noise, cfg.seed, deterministic=False) if cfg.noise != "none" else None)
    val_ds = ModelReadyDataset(val_eps, bcfg, transform=ImuNoise(cfg.val_noise, 12345) if cfg.val_noise != "none" else None)
    log(f"train: {len(train_eps)} episodes, {len(train_ds)} windows; val: {len(val_eps)} episodes, {len(val_ds)} windows; device {device}")
    ident = dataset_identity(dataset_dir)

    state = None
    last = out_dir / "checkpoints" / "last.pt"
    if resume and last.exists():
        state = ckpt.load(last)
        if state["dataset"] != ident and not force:
            raise RuntimeError(f"checkpoint was trained on {state['dataset']}, now {ident}; pass force=True to override")
        if state["model_cfg"] != asdict(model_cfg_of(cfg)) and not force:
            raise RuntimeError("model configuration differs from the checkpoint")
        model = build_from_checkpoint(state); weights = LossWeights(**state["loss_weights"])
        log(f"resuming from epoch {state['epoch']} (step {state['step']})")
    else:
        stats = fit_stats(train_eps)
        model = build_model(cfg.kind, stats=stats, cfg=model_cfg_of(cfg), dt=train_ds.episodes[0].dt)
        weights = fit_loss_weights(train_ds, min(20, P), gamma=cfg.gamma, lambda_roll=cfg.lambda_roll, seed=cfg.seed)
    model.to(device)
    log(f"loss weights: {weights.to_dict()}")
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_ds) // cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total_steps, cfg.warmup_steps))

    epoch, step, stage, bad, stage_epochs, best = 0, 0, 0, 0, 0, float("inf")
    if state is not None:
        opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        epoch, step, stage, bad, stage_epochs, best = (state[k] for k in ("epoch", "step", "stage", "bad", "stage_epochs", "best"))
        ckpt.set_rng_state(state["rng"])
    else:
        (out_dir / "config.yaml").write_text(json.dumps({"train": asdict(cfg), "builder": asdict(bcfg)}, indent=2))
        (out_dir / "provenance.json").write_text(json.dumps(
            {"git": _git_commit(), "torch": torch.__version__, "dataset": ident, "n_params": n_params,
             "started": time.strftime("%Y-%m-%d %H:%M:%S"), "loss_weights": weights.to_dict()}, indent=2))

    tracker = Tracker(out_dir, out_dir.name, {"model": cfg.kind, "dataset": f"{ident.get('dataset_id')}_v{ident.get('dataset_version')}",
                                              "n_params": n_params, "git": _git_commit(), "noise": cfg.noise, "resumed": state is not None},
                      use_tensorboard=cfg.use_tensorboard)
    val_k = min(20, P)
    metrics_path = out_dir / "metrics.jsonl"
    done = False
    stop_reason = "max_epochs"
    while epoch < cfg.epochs and not done:
        h = int(cfg.horizon_stages[stage])
        model.train()
        t0, agg, nb = time.time(), {}, 0
        for batch in make_loader(train_ds, cfg, epoch):
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
        if len(val_ds):
            ev = evaluate(model, val_ds, horizon=val_k, device=device, batch_size=512, initial_state=cfg.initial_state, n_boot=0)
            rec["val"] = {"pos_m": ev["pos_m"], "att_deg": ev["att_deg"], "reliable_horizon": ev["reliable_horizon_mean"]}
            metric = ev["pos_m"][str(val_k)]
        else:
            metric = rec["train"]["total"]
        improved = metric < best - 1e-12
        if improved:
            best, bad = metric, 0
        else:
            bad += 1
        rec["metric"], rec["best"] = metric, best
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
                "rng": ckpt.rng_state(), "stats": {k: getattr(model, k).tolist() for k in STAT_KEYS},
                "kind": cfg.kind, "model_cfg": asdict(model_cfg_of(cfg)), "dt": model.dt,
                "loss_weights": weights.to_dict(), "dataset": ident, "train_cfg": asdict(cfg), "git": _git_commit(),
                "metric": metric}
        ckpt.save_atomic(snap, last)
        ckpt.save_atomic(snap, out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")
        if improved:
            ckpt.save_atomic(snap, out_dir / "checkpoints" / "best.pt")
        ckpt.prune(out_dir / "checkpoints", cfg.keep_epoch_checkpoints)
        if max_epochs is not None and epoch >= max_epochs:
            break
    else:
        stop_reason = "epochs_done" if epoch >= cfg.epochs else stop_reason
    summary = {"epochs": epoch, "best_metric": best, "stop_reason": stop_reason, "n_params": n_params, "out_dir": str(out_dir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tracker.end({**summary, "model": cfg.kind})
    return summary
