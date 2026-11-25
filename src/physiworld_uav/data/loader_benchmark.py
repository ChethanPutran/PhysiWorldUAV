"""Data-loading throughput benchmark (e2e requirement: "test whether the training pipeline can consume the data fast enough").

    python -m physiworld_uav.data.loader_benchmark --dataset data/datasets/physiworld_pilot_v0.5.0 --workers 0 2 4 8 \\
        --run outputs/experiments/pipeline_smoke_b3 --out outputs/eval/loader_benchmark.json

Measures (1) reading + decimating the episodes from HDF5 (episodes/s, MB/s of file), (2) random-window batches through a
``DataLoader`` for several ``num_workers`` (windows/s), and (3) the headroom against the training consumption rate (windows/s
logged by the trainer in ``<run>/metrics.jsonl``): headroom < 1 means the GPU would wait for data.
Window sampling cuts views of in-memory arrays, so the cost of one window is small; the numbers show whether collation and
(optional) noise injection are the limit.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ModelReadyDataset
from .dataset_builder import BuilderConfig, dataset_decimation, load_model_episodes
from .noise import ImuNoise


def time_loader(ds, batch_size: int, num_workers: int, n_batches: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g,
                        drop_last=len(ds) >= batch_size, persistent_workers=num_workers > 0)
    it = iter(loader)
    next(it)                                                     # warm-up: worker start-up, first batch
    t0, n = time.perf_counter(), 0
    for _ in range(n_batches):
        try:
            b = next(it)
        except StopIteration:
            break
        n += int(b["state_history"].shape[0])
    dt = time.perf_counter() - t0
    return {"num_workers": num_workers, "windows": n, "seconds": dt, "windows_per_s": n / max(dt, 1e-9)}


def training_rate(run_dir: Path | None) -> float | None:
    """Median windows/s the trainer logged (None without a run)."""
    if run_dir is None or not (Path(run_dir) / "metrics.jsonl").exists():
        return None
    rec = [json.loads(l) for l in (Path(run_dir) / "metrics.jsonl").read_text().splitlines() if l.strip()]
    rates = [r["windows_per_s"] for r in rec if "windows_per_s" in r]
    return float(np.median(rates)) if rates else None


def benchmark(dataset_dir: str | Path, *, roles: Sequence[str] = ("train",), history: int = 10, horizon: int = 20,
              batch_size: int = 256, workers: Sequence[int] = (0, 2, 4), n_batches: int = 40, noise: str = "nominal",
              run_dir: Path | None = None, episodes=None) -> dict:
    d = Path(dataset_dir)
    bcfg = BuilderConfig(decimation=dataset_decimation(d), history=history, horizon=horizon)
    t0 = time.perf_counter()
    eps = list(episodes) if episodes is not None else load_model_episodes(d, bcfg, roles=list(roles))
    t_load = time.perf_counter() - t0
    shard_bytes = sum(p.stat().st_size for p in d.glob("shard_*.h5")) if d.exists() else 0
    ds = ModelReadyDataset(eps, bcfg, transform=ImuNoise(noise, 0, deterministic=False) if noise != "none" else None)
    out = {"dataset": str(d), "roles": list(roles), "n_episodes": len(eps), "n_windows": len(ds), "batch_size": batch_size,
           "load_episodes": {"seconds": t_load, "episodes_per_s": len(eps) / max(t_load, 1e-9),
                             "shard_MB": shard_bytes / 1e6, "MB_per_s": shard_bytes / 1e6 / max(t_load, 1e-9)},
           "loader": [time_loader(ds, batch_size, w, n_batches) for w in workers]}
    rate = training_rate(run_dir)
    if rate:
        out["training_windows_per_s"] = rate
        best = max(r["windows_per_s"] for r in out["loader"])
        out["headroom_best_loader_over_training"] = best / rate
    return out


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(prog="loader_benchmark", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--roles", nargs="*", default=["train"])
    p.add_argument("--history", type=int, default=10)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", nargs="*", type=int, default=[0, 2, 4])
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--noise", default="nominal")
    p.add_argument("--run", type=Path, default=None, help="training run directory (reads windows_per_s from metrics.jsonl)")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    res = benchmark(a.dataset, roles=a.roles, history=a.history, horizon=a.horizon, batch_size=a.batch_size, workers=a.workers,
                    n_batches=a.batches, noise=a.noise, run_dir=a.run)
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
