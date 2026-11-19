"""CLI: train a baseline on a dataset.

    python -m physiworld_uav.training.run --dataset data/datasets/physiworld_pilot_v0.5.0 --name b2_pilot
    python -m physiworld_uav.training.run --dataset ... --name g1 --tiny 8          # G1 tiny-overfit test
    python -m physiworld_uav.training.run --dataset ... --name b2_pilot --resume
    python -m physiworld_uav.training.run ... --set epochs=50 lr=3e-4 horizon_stages=[1,5,10,20]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .trainer import TrainConfig, train


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(prog="train", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--name", required=True, help="run name; output in outputs/experiments/<name>")
    p.add_argument("--model", default="b2", help="b2 | b3 | b4 (b5, b6 not implemented yet)")
    p.add_argument("--out-root", type=Path, default=Path("outputs/experiments"))
    p.add_argument("--tiny", type=int, default=0, help="overfit test: train on the first N episodes, no validation")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true", help="resume although dataset/config differ")
    p.add_argument("--device", default="auto")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="TrainConfig overrides (JSON values)")
    a = p.parse_args(argv)
    cfg = TrainConfig(kind=a.model, tiny_episodes=a.tiny, device=a.device)
    for kv in a.set:
        k, v = kv.split("=", 1)
        if not hasattr(cfg, k):
            p.error(f"unknown config key {k!r}")
        setattr(cfg, k, json.loads(v) if v[:1] in "[{0123456789-" or v in ("true", "false") else v)
    summary = train(a.dataset, a.out_root / a.name, cfg, resume=a.resume, force=a.force)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
