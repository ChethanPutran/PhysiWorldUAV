"""Configuration composition (plan §3-C1).

* Top level: Hydra composes ``config/defaults.yaml`` with its defaults list
  (drone, physics, limits, controller, dataset, experiment) plus command-line
  overrides.
* Templates referenced *by name* from inside an experiment (scenario,
  wind regime, domain, excitation, initial conditions, counterfactual,
  forces) are loaded with ``load_template(group, name, overrides)``, which
  merges per-use overrides with OmegaConf. Experiments therefore reference
  shared templates and override only what differs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"

TEMPLATE_GROUPS = ("scenario", "wind/regimes", "domain", "excitation", "initial_conditions",
                   "counterfactual", "forces", "drone", "limits", "controller", "physics", "dataset",
                   "experiment")


def compose(overrides: Sequence[str] = (), config_name: str = "defaults",
            config_dir: str | Path = CONFIG_DIR) -> DictConfig:
    """Compose the root config with Hydra (no working-directory change, no logging setup)."""
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base="1.3"):
        cfg = hydra_compose(config_name=config_name, overrides=list(overrides))
    return cfg


def load_template(group: str, name: str, overrides: Mapping[str, Any] | None = None,
                  config_dir: str | Path = CONFIG_DIR) -> dict[str, Any]:
    if group not in TEMPLATE_GROUPS:
        raise ValueError(f"unknown template group {group!r}")
    path = Path(config_dir) / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"template {group}/{name} not found at {path}")
    base = OmegaConf.load(path)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.create(dict(overrides)))
    out = OmegaConf.to_container(base, resolve=True)
    if not isinstance(out, dict):
        raise ValueError(f"template {group}/{name} must be a mapping")
    out.setdefault("name", name)
    return out


def to_plain(cfg: Any) -> Any:
    """Plain Python containers from a DictConfig (or pass-through)."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg
