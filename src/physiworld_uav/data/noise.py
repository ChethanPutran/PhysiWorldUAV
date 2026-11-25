"""IMU measurement noise, applied at **load time** (decision D7): the stored IMU is clean, so one dataset
serves several noise levels.

Model per window (the history is 0.2 s, so the bias is constant within a window):

    o_obs = o_clean + b + n,     b ~ N(0, σ_bias²) per channel and window,   n ~ N(0, σ_white²) per sample and channel

Channels: accelerometer (m/s², body) and gyroscope (rad/s, body). The presets are PROPOSED values of the order of
consumer-to-tactical MEMS IMUs at a 50 Hz output (white-noise σ of the 50 Hz samples, not a noise density); they are
configuration, not truth, and should be reported with every result. A fixed ``seed`` makes evaluation reproducible:
the noise of a window depends only on (seed, episode id, time index).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch

PRESETS = {                                   # accel σ_white, accel σ_bias, gyro σ_white, gyro σ_bias
    "none":    (0.0, 0.0, 0.0, 0.0),
    "low":     (0.05, 0.02, 0.005, 0.002),
    "nominal": (0.10, 0.05, 0.010, 0.005),
    "high":    (0.20, 0.10, 0.020, 0.010),
}


@dataclass(frozen=True)
class ImuNoiseConfig:
    accel_white: float = 0.0
    accel_bias: float = 0.0
    gyro_white: float = 0.0
    gyro_bias: float = 0.0

    @classmethod
    def preset(cls, name: str) -> "ImuNoiseConfig":
        if name not in PRESETS:
            raise ValueError(f"unknown noise preset {name!r}; choose from {sorted(PRESETS)}")
        return cls(*PRESETS[name])

    @property
    def enabled(self) -> bool:
        return any((self.accel_white, self.accel_bias, self.gyro_white, self.gyro_bias))


class ImuNoise:
    """Dataset transform: adds noise to ``obs_history`` only (states and actions are untouched)."""

    def __init__(self, cfg: ImuNoiseConfig | str, seed: int = 0, deterministic: bool = True) -> None:
        self.cfg = ImuNoiseConfig.preset(cfg) if isinstance(cfg, str) else cfg
        self.seed, self.deterministic = int(seed), deterministic

    def _generator(self, sample: dict) -> torch.Generator | None:
        if not self.deterministic:
            return None
        key = zlib.crc32(f"{self.seed}|{sample.get('episode_id', '')}|{int(sample.get('time_index', 0))}".encode())
        return torch.Generator().manual_seed(key)

    def __call__(self, sample: dict) -> dict:
        if not self.cfg.enabled:
            return sample
        obs = sample["obs_history"]
        g = self._generator(sample)
        sig_w = torch.tensor([self.cfg.accel_white] * 3 + [self.cfg.gyro_white] * 3, dtype=obs.dtype)
        sig_b = torch.tensor([self.cfg.accel_bias] * 3 + [self.cfg.gyro_bias] * 3, dtype=obs.dtype)
        white = torch.randn(obs.shape, generator=g, dtype=obs.dtype) * sig_w
        bias = torch.randn(obs.shape[-1], generator=g, dtype=obs.dtype) * sig_b
        out = dict(sample)
        out["obs_history"] = obs + white + bias
        if "future_obs" in sample:        # re-observation analyses: same bias (constant within the window), fresh white noise
            fo = sample["future_obs"]
            out["future_obs"] = fo + torch.randn(fo.shape, generator=g, dtype=fo.dtype) * sig_w + bias
        return out
