"""Atomic checkpoints with everything needed to resume (docs/specs/training_pipeline.md §4)."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"]); np.random.set_state(s["numpy"]); random.setstate(s["python"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_atomic(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def prune(directory: str | Path, keep: int = 3) -> None:
    files = sorted(Path(directory).glob("epoch_*.pt"))
    for f in files[:-keep]:
        f.unlink()
