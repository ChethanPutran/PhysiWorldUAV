"""Model-ready episodes at the model rate (plan §6.5-DS1/DS3/DS4/DS6, D3, D4).

A ``ModelEpisode`` is one episode decimated to the model step Δt = D·dt with:

* ``states``  (M+1, 13)  p_W | v_W | q_wxyz | ω_B              (ground truth)
* ``obs``     (M+1, 6)   IMU specific force | IMU gyro          (observations, D7)
* ``actions`` (M, A)     action held/applied over [t_j, t_{j+1})
* ``aux``     dict       wind / wrench at the model rate         (never a model input)

Action keys (D3; configurable via ``dataset.action_key``):

* ``rotor_speed_mean`` (default) — mean of Ω over the physics instants inside
  the interval (captures the motor lag response to the held command)
* ``rotor_speed``      — Ω at the interval start
* ``rotor_cmd``        — applied OmniDrones command (exact zero-order hold)
* ``rotor_speed_cmd``  — steady-state Ω target of the command
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .metadata import EpisodeMetadata
from .resampler import decimate_episode
from .schema import MODEL_OBS_FIELDS, MODEL_STATE_FIELDS, MODEL_STATE_LAYOUT
from .writer import read_episode

ACTION_KEYS = {
    "rotor_speed_mean": ("ground_truth/rotor_speed__mean", "mean"),
    "rotor_speed": ("ground_truth/rotor_speed", "start"),
    "rotor_cmd": ("actions/rotor_cmd", "held"),
    "rotor_speed_cmd": ("actions/rotor_speed_cmd", "held"),
}
AUX_FIELDS = ("environment/wind_W", "environment/wind_applied_W", "environment/wind_rotor_W",
              "ground_truth/wrench/external_force_W", "ground_truth/wrench/external_torque_B")


def dataset_sim(dataset_dir) -> dict:
    """``{"control_decimation": int, "model_dt": float}`` of a dataset, from the first row of its manifest (the plan fixes them for every
    episode). A dataset or test fixture without a manifest gets the 50 Hz defaults (decimation 4, 0.02 s)."""
    import json
    from pathlib import Path
    try:
        with (Path(dataset_dir) / "manifest.jsonl").open() as f:
            sim = json.loads(f.readline())["sim"]
        return {"control_decimation": int(sim["control_decimation"]), "model_dt": float(sim["model_dt"])}
    except (OSError, KeyError, ValueError, TypeError):
        return {"control_decimation": 4, "model_dt": 0.02}


def dataset_decimation(dataset_dir) -> int:
    return dataset_sim(dataset_dir)["control_decimation"]


@dataclass(frozen=True)
class BuilderConfig:
    decimation: int = 4
    history: int = 10                 # H
    horizon: int = 20                 # P
    stride: int = 1
    action_key: str = "rotor_speed_mean"
    obs_mode: str = "instant"         # instant | interval_mean
    statuses: tuple[str, ...] = ("completed",)
    exclude_open_loop_kinds: tuple[str, ...] = ("body_wrench",)

    def __post_init__(self) -> None:
        if self.action_key not in ACTION_KEYS:
            raise ValueError(f"action_key must be one of {sorted(ACTION_KEYS)}")
        if self.obs_mode not in ("instant", "interval_mean"):
            raise ValueError("obs_mode must be 'instant' or 'interval_mean'")
        if self.history < 1 or self.horizon < 1 or self.stride < 1:
            raise ValueError("history, horizon and stride must be >= 1")

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any], decimation: int) -> "BuilderConfig":
        from physiworld_uav.core.config_loader import cfg_get
        return cls(
            decimation=int(decimation),
            history=int(cfg_get(cfg, "history", 10)),
            horizon=int(cfg_get(cfg, "horizon", 20)),
            stride=int(cfg_get(cfg, "stride", 1)),
            action_key=str(cfg_get(cfg, "action_key", "rotor_speed_mean")),
            obs_mode=str(cfg_get(cfg, "obs_mode", "instant")),
            statuses=tuple(cfg_get(cfg, "statuses", ("completed",))),
            exclude_open_loop_kinds=tuple(cfg_get(cfg, "exclude_open_loop_kinds", ("body_wrench",))),
        )


@dataclass
class ModelEpisode:
    episode_id: str
    meta: EpisodeMetadata
    t: np.ndarray
    states: np.ndarray
    obs: np.ndarray
    actions: np.ndarray
    dt: float
    aux: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.actions.shape[0])


def to_model_episode(arrays: Mapping[str, np.ndarray], meta: EpisodeMetadata,
                     cfg: BuilderConfig) -> ModelEpisode:
    dec = decimate_episode(arrays, cfg.decimation)
    states = np.concatenate([dec[p] for p, _ in MODEL_STATE_FIELDS], axis=1).astype(np.float32)
    if cfg.obs_mode == "instant":
        obs = np.concatenate([dec[p] for p, _ in MODEL_OBS_FIELDS], axis=1)
    else:   # mean over the interval *preceding* each instant; first row = instant value
        means = [dec[p + "__mean"] for p, _ in MODEL_OBS_FIELDS]
        first = [dec[p][:1] for p, _ in MODEL_OBS_FIELDS]
        obs = np.concatenate([np.concatenate([f, m], axis=0) for f, m in zip(first, means)], axis=1)
    key, kind = ACTION_KEYS[cfg.action_key]
    act = dec[key]
    if kind == "start":
        act = act[:-1]
    t = dec["time/t"]
    m = states.shape[0] - 1
    if act.shape[0] != m:
        raise ValueError(f"{meta.episode_id}: {act.shape[0]} actions for {m} model steps")
    aux = {p: dec[p] for p in AUX_FIELDS if p in dec}
    return ModelEpisode(episode_id=meta.episode_id, meta=meta, t=t, states=states,
                        obs=obs.astype(np.float32), actions=act.astype(np.float32),
                        dt=float(t[1] - t[0]), aux=aux)


def imu_transport_status(episodes: Sequence) -> str:
    """"corrected" | "raw" | "mixed" | "unknown": was the Isaac IMU transport artefact (−ω×v) removed at generation?
    Datasets from v0.6.0 on are corrected; the Run 5 pilot v0.5.0 holds the raw sensor (manifest rows have no flag)."""
    flags = {bool((e.meta.sim or {}).get("imu_transport_correction", False)) for e in episodes if getattr(e, "meta", None) is not None}
    if not flags:
        return "unknown"
    return "corrected" if flags == {True} else ("raw" if flags == {False} else "mixed")


IMU_RAW_WARNING = ("this dataset holds the RAW Isaac IMU (accelerometer = specific force − ω×v): the synthetic IMU φ of the latent models "
                   "produces the true specific force and will not match it. Use a dataset generated with physics.imu_transport_correction=true (v0.6.0+).")


def _eligible(meta: EpisodeMetadata, cfg: BuilderConfig, roles: Sequence[str] | None) -> bool:
    if meta.status not in cfg.statuses:
        return False
    if roles is not None and meta.split_role not in roles:
        return False
    if not meta.stats.get("applicability_ok", True):      # plan F11: model used outside its envelope
        return False
    ol = meta.plan.get("open_loop") or {}
    return ol.get("kind") not in cfg.exclude_open_loop_kinds


def load_index(dataset_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(dataset_dir) / "index.json").read_text())


def load_model_episodes(dataset_dir: str | Path, cfg: BuilderConfig,
                        roles: Sequence[str] | None = None) -> list[ModelEpisode]:
    dataset_dir = Path(dataset_dir)
    index = load_index(dataset_dir)
    out = []
    for row in index["episodes"]:
        if roles is not None and row["split_role"] not in roles:
            continue
        arrays, meta = read_episode(dataset_dir / row["shard"], row["episode_id"])
        if _eligible(meta, cfg, roles):
            out.append(to_model_episode(arrays, meta, cfg))
    return out


def window_starts(n_steps: int, cfg: BuilderConfig) -> np.ndarray:
    """Valid current-time indices t with H states and H preceding actions behind,
    and P future actions/states ahead: t ∈ [H, M − P]."""
    lo, hi = cfg.history, n_steps - cfg.horizon
    if hi < lo:
        return np.zeros(0, dtype=np.int64)
    return np.arange(lo, hi + 1, cfg.stride, dtype=np.int64)


def model_layout() -> dict[str, Any]:
    return {"state": list(MODEL_STATE_LAYOUT),
            "obs": ["imu_accel_B[0:3]", "imu_gyro_B[3:6]"],
            "action_keys": sorted(ACTION_KEYS)}
