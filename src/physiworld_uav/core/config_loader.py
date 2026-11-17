"""Small, side-effect-free YAML config loader. Used by all Phase 2+ modules."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[3]   # .../PhysiWorldUAV


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    return p


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a plain dict, an OmegaConf DictConfig, or any object.

    ``getattr`` alone silently returns ``default`` for plain dicts (what
    ``load_yaml`` returns), so config sections would be ignored.
    """
    if cfg is None:
        return default
    if isinstance(cfg, dict) or hasattr(cfg, "get"):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value is None else value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"top-level YAML in {path} must be a mapping")
    return data


def load_hummingbird(path: str | Path = "config/drone/hummingbird.yaml") -> dict[str, Any]:
    cfg = load_yaml(path)
    # Hard consistency checks that guard against silent divergence.
    assert cfg["name"] == "hummingbird"
    assert abs(cfg["arm_length_diagonal"] - cfg["l"]) < 1e-9
    assert abs(cfg["arm_length_individual"] - cfg["l"] / 2.0) < 1e-9
    assert cfg["rotor_configuration"]["num_rotors"] == 4
    assert len(cfg["rotor_configuration"]["arm_lengths"]) == 4
    return cfg