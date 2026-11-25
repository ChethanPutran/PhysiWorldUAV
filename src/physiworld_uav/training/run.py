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

from .config_io import build_dataclass, merged
from .trainer import TrainConfig, train


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(prog="train", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--name", required=True, help="run name; output in outputs/experiments/<name>")
    p.add_argument("--model", default=None, help="b2 | b3 | b4 (matched total) | b4_rw (B3 residual width) | b5a | b5b | b6b_future | b6b_current (default: `kind` from --config, else b2)")
    p.add_argument("--out-root", type=Path, default=Path("outputs/experiments"))
    p.add_argument("--tiny", type=int, default=0, help="overfit test: train on the first N episodes, no validation")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true", help="resume although dataset/config differ")
    p.add_argument("--device", default="auto")
    p.add_argument("--config", nargs="*", default=[], metavar="YAML", help="YAML files merged left to right (config/training/*.yaml, config/model/*.yaml)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="TrainConfig overrides (JSON values), applied after the YAML files")
    a = p.parse_args(argv)
    values = merged(a.config, a.set)
    if a.model is not None:                       # an explicit --model wins over the YAML; --set kind=... still wins over both
        values.setdefault("kind", a.model) if any(kv.startswith("kind=") for kv in a.set) else values.update(kind=a.model)
    values.setdefault("kind", "b2")
    if a.tiny:
        values["tiny_episodes"] = a.tiny
    if a.device != "auto" or "device" not in values:
        values["device"] = a.device
    try:
        cfg = build_dataclass(TrainConfig, values)
    except ValueError as e:
        p.error(str(e))
    summary = train(a.dataset, a.out_root / a.name, cfg, resume=a.resume, force=a.force)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
