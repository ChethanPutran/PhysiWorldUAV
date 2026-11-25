"""Torch datasets over model-ready episodes (plan §6.5).

(``SyntheticUAVDataset`` moved to ``lagacy/synthetic_dataset.py``.)

Sample semantics at current model step t (H = history, P = horizon):

* ``state_history``  (H, 13)  s_{t−H+1 … t}
* ``obs_history``    (H, 6)   o_{t−H+1 … t}            (IMU)
* ``action_history`` (H, A)   u_{t−H … t−1}            (u_k produced s_{k+1})
* ``future_actions`` (P, A)   u_{t … t+P−1}
* ``future_states``  (P, 13)  s_{t+1 … t+P}            (s_{t+k} results from u_{t…t+k−1})
* ``dt``             model step (s)

Windows are cut lazily from contiguous episode arrays (no materialised
copies). Auxiliary data (wind, injected wrench) is **not** part of the
primary sample; ``AuxiliaryView`` exposes it for evaluation only.

Exception (B6 only): ``ModelReadyDataset(..., oracle=True)`` adds ``oracle_force_W`` / ``oracle_torque_B`` (P, 3),
the recorded **total injected aero wrench** acting during steps t … t+P−1 (decision: disturbance target = total wrench).
It is off by default and is switched on only for the B6 oracle models, which are reference upper bounds.
``future_obs=True`` adds ``future_obs`` (P, 6): the real IMU at t+1 … t+P, for evaluation analyses that re-observe the
vehicle every K steps (``evaluation/analysis.py``). Never used in training.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_builder import BuilderConfig, ModelEpisode, window_starts
from .schema import AUXILIARY_PREFIXES


ORACLE_SRC = ("ground_truth/wrench/external_force_W", "ground_truth/wrench/external_torque_B")
ORACLE_KEYS = ("oracle_force_W", "oracle_torque_B")


class ModelReadyDataset(Dataset):
    def __init__(self, episodes: Sequence[ModelEpisode], cfg: BuilderConfig,
                 transform: Optional[Callable[[dict], dict]] = None, oracle: bool = False,
                 future_obs: bool = False) -> None:
        self.episodes = list(episodes)
        self.oracle = bool(oracle)
        self.future_obs = bool(future_obs)
        if self.oracle:
            missing = [e.episode_id for e in self.episodes if ORACLE_SRC[0] not in e.aux or ORACLE_SRC[1] not in e.aux]
            if missing:
                raise ValueError(f"oracle=True needs the recorded wrench in every episode; missing in {missing[:3]}")
        self.cfg = cfg
        self.transform = transform
        index = [(e, int(t)) for e, ep in enumerate(self.episodes) for t in window_starts(ep.n_steps, cfg)]
        self._index = np.asarray(index, dtype=np.int64).reshape(-1, 2)
        dts = {round(ep.dt, 9) for ep in self.episodes}
        if len(dts) > 1:
            raise ValueError(f"episodes have different model dt: {sorted(dts)}")

    def __len__(self) -> int:
        return int(self._index.shape[0])

    def window(self, i: int) -> tuple[ModelEpisode, int]:
        e, t = self._index[i]
        return self.episodes[int(e)], int(t)

    def __getitem__(self, i: int) -> dict:
        ep, t = self.window(i)
        H, P = self.cfg.history, self.cfg.horizon
        sample = {
            "state_history": torch.from_numpy(ep.states[t - H + 1: t + 1]),
            "obs_history": torch.from_numpy(ep.obs[t - H + 1: t + 1]),
            "action_history": torch.from_numpy(ep.actions[t - H: t]),
            "future_actions": torch.from_numpy(ep.actions[t: t + P]),
            "future_states": torch.from_numpy(ep.states[t + 1: t + P + 1]),
            "dt": torch.tensor(ep.dt, dtype=torch.float32),
            "episode_id": ep.episode_id,
            "time_index": t,
        }
        if self.future_obs:          # evaluation only: the real IMU at t+1 … t+P (re-observation / boundary analyses)
            sample["future_obs"] = torch.from_numpy(np.asarray(ep.obs[t + 1: t + P + 1]))
        if self.oracle:
            sample["oracle_force_W"] = torch.from_numpy(np.asarray(ep.aux[ORACLE_SRC[0]][t: t + P], dtype=np.float32))
            sample["oracle_torque_B"] = torch.from_numpy(np.asarray(ep.aux[ORACLE_SRC[1]][t: t + P], dtype=np.float32))
        if self.transform is not None:
            sample = self.transform(sample)
        leaked = [k for k in sample if k.startswith(AUXILIARY_PREFIXES)]
        if leaked:
            raise RuntimeError(f"auxiliary keys in primary sample: {leaked}")
        return sample


class AuxiliaryView:
    """Evaluation-only access to wind / wrench / parameters for a sample."""

    def __init__(self, dataset: ModelReadyDataset) -> None:
        self._ds = dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int) -> dict:
        ep, t = self._ds.window(i)
        P = self._ds.cfg.horizon
        out = {k: torch.from_numpy(np.asarray(v[t: t + P + 1])) for k, v in ep.aux.items()}
        out.update(episode_id=ep.episode_id, drone=ep.meta.drone,
                   wind_plan=ep.meta.plan.get("wind"), split_role=ep.meta.split_role)
        return out
