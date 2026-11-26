"""Example: plan, generate, validate and load a dataset (schema v2).

Run inside the isaac-sim container with the venv activated:

    export PYTHONPATH=$PWD/src:$PYTHONPATH
    python examples/example0_generate_data.py                  # smoke experiment
    python examples/example0_generate_data.py experiment=pilot plan_only=true

Equivalent CLI:  python -m physiworld_uav.experiments.generate run experiment=smoke

The previous 950-line example was removed with the superseded code (archive: /data2/chethan/mtech/archive/lagacy_2026-11-26.tar.gz).
"""
from __future__ import annotations

import logging
import sys

from physiworld_uav.config import compose, to_plain
from physiworld_uav.experiments.generate import dataset_dir, run, write_plan


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    plan_only = "plan_only=true" in argv
    overrides = [a for a in argv if a != "plan_only=true"] or ["experiment=smoke"]
    cfg = compose(overrides)

    if plan_only:
        out, manifest, _ = write_plan(cfg)
        print(f"planned {len(manifest.episodes)} episodes → {out}: {manifest.by_role()}")
        return

    # 1. plan → 2. simulate (Isaac) → 3. validate → index.json; Isaac closes the process at the end.
    out = dataset_dir(to_plain(cfg))
    print(f"writing to {out}")
    run(cfg)


def load_example(dataset: str) -> None:
    """How to read the model-ready view (CPU, no Isaac needed)."""
    from physiworld_uav.data.dataset import ModelReadyDataset
    from physiworld_uav.data.dataset_builder import BuilderConfig, load_model_episodes

    cfg = BuilderConfig(decimation=4, history=10, horizon=20)
    episodes = load_model_episodes(dataset, cfg, roles=["train"])
    ds = ModelReadyDataset(episodes, cfg)
    sample = ds[0]
    print({k: getattr(v, "shape", v) for k, v in sample.items()})


if __name__ == "__main__":
    main(sys.argv[1:])
