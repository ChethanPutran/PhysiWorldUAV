"""YAML configuration for training and evaluation (``config/training``, ``config/model``, ``config/evaluation``).

A run is described by YAML files merged left to right (later wins) and finally by ``--set key=value`` overrides::

    python -m physiworld_uav.training.run --dataset D --name r1 --config config/training/default.yaml config/model/b3.yaml \\
        --set epochs=5

Unknown keys are an error (a typo must not silently fall back to a default). The resolved configuration is stored in the run
directory (``config.yaml``). The YAML defaults are kept identical to the dataclass defaults by a test.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    d = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{path}: a configuration file must be a mapping")
    return d


def parse_override(kv: str) -> tuple[str, Any]:
    k, v = kv.split("=", 1)
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v                                   # plain string (e.g. noise=nominal)


def merged(files: Iterable[str | Path] = (), overrides: Iterable[str] = ()) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in files:
        out.update(load_yaml(f))
    for kv in overrides:
        k, v = parse_override(kv)
        out[k] = v
    return out


def build_dataclass(cls, values: Mapping[str, Any]):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} key(s) {unknown}; valid keys: {sorted(names)}")
    return cls(**values)


def train_config(files: Iterable[str | Path] = (), overrides: Iterable[str] = (), **fixed):
    from .trainer import TrainConfig
    return build_dataclass(TrainConfig, {**merged(files, overrides), **fixed})


EVAL_KEYS = {"horizon", "noise", "initial_state", "roles", "n_boot", "every", "mode"}


def eval_defaults(path: str | Path | None) -> dict[str, Any]:
    """Defaults for the evaluation CLIs from a YAML file (keys restricted to ``EVAL_KEYS``)."""
    if path is None:
        return {}
    d = load_yaml(path)
    unknown = sorted(set(d) - EVAL_KEYS)
    if unknown:
        raise ValueError(f"unknown evaluation key(s) {unknown}; valid keys: {sorted(EVAL_KEYS)}")
    return d
