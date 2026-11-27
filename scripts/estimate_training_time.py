#!/usr/bin/env python3
"""Estimate the training time of a larger dataset (e.g. Run 6) from measured runs (outputs/experiments/<run>/metrics.jsonl).

    python3 scripts/estimate_training_time.py --runs v06_b2 v06_b3 ... [--train-episodes 190 630 1900] [--md out.md]

Measured per run: epochs, training seconds (epoch_s: forward/backward over all training windows of one epoch), validation seconds
(wall_s - epoch_s when the run recorded resources, else not available), number of training episodes (from the log line, or
--pilot-train-episodes). The model is linear and deliberately simple:

    training time per epoch  ∝ number of training windows ∝ number of training episodes (same episode lengths assumed)
    validation time per epoch ∝ number of validation episodes (the pilot ratio val/train is kept)
    number of epochs          = as measured (ASSUMPTION: a larger dataset may converge in fewer or more epochs)

so   T(N) = T_pilot × N / N_pilot.  It ignores that the host was shared (up to 9 jobs on 3 GPUs while measuring) and that more
episodes need more CPU memory (episodes are held in RAM: peak RSS grows roughly linearly with the number of episodes; the
pilot's ≈ 1 GB is dominated by the 63 training + validation episodes and the CUDA/torch runtime).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_stats(root: Path, name: str, pilot_n: int) -> dict | None:
    f = root / name / "metrics.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        return None
    train = sum(r["epoch_s"] for r in rows)
    val = sum(max((r.get("resources") or {}).get("wall_s", r["epoch_s"]) - r["epoch_s"], 0.0) for r in rows) if "resources" in rows[0] else None
    n_eps = rows[0].get("n_train_episodes", pilot_n)
    return {"run": name, "epochs": len(rows), "train_s": train, "val_s": val, "n_train_episodes": n_eps,
            "s_per_episode_epoch": train / len(rows) / n_eps, "peak_rss_mb": max((r.get("resources") or {}).get("rss_peak_mb") or 0 for r in rows),
            "windows_per_s": sum(r["windows_per_s"] for r in rows) / len(rows)}


def hms(s: float) -> str:
    s = int(s); return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min {s % 60:02d} s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--exp-root", type=Path, default=Path("outputs/experiments"))
    ap.add_argument("--train-episodes", type=int, nargs="+", default=[190, 630, 1900], help="training episodes of the larger dataset (scenarios)")
    ap.add_argument("--pilot-train-episodes", type=int, default=63)
    ap.add_argument("--epoch-factor", type=float, default=1.0, help="assumed (epochs on the large dataset) / (epochs measured)")
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()
    st = [s for s in (run_stats(a.exp_root, r, a.pilot_train_episodes) for r in a.runs) if s]
    out = ["| Run | Train episodes | Epochs | Training time (measured) | s per episode per epoch | Windows/s | " +
           " | ".join(f"{n} train episodes" for n in a.train_episodes) + " |",
           "|---|---|---|---|---|---|" + "---|" * len(a.train_episodes)]
    for s in st:
        est = [hms((s["train_s"] + (s["val_s"] or 0) * 0) * n / s["n_train_episodes"] * a.epoch_factor) for n in a.train_episodes]
        out.append(f"| {s['run']} | {s['n_train_episodes']} | {s['epochs']} | {hms(s['train_s'])} | {s['s_per_episode_epoch']:.3f} | {s['windows_per_s']:.0f} | " + " | ".join(est) + " |")
    text = "\n".join(out)
    text += ("\n\nTraining time only (validation excluded; the new runs record it separately as `val_s`). Linear in the number of "
             "training episodes, epochs as measured × " + f"{a.epoch_factor:g}" + "; measured with up to 9 jobs sharing 3 GPUs.\n")
    print(text)
    if a.md:
        a.md.write_text(text)


if __name__ == "__main__":
    main()
