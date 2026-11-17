"""Episode recorder with preallocated device buffers (plan §4, §6.8-PF1).

Values are written into tensors on the simulation device; the host copy
happens once, in ``to_numpy``. Batched over environments: every buffer has
shape (E, rows, *width).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .schema import BY_PATH, expected_rows


class EpisodeRecorder:
    def __init__(self, n_steps: int, fields: Iterable[str], *, n_envs: int = 1,
                 device: str | torch.device = "cpu") -> None:
        self.n_steps = int(n_steps)
        self.n_envs = int(n_envs)
        self.device = torch.device(device)
        self._buf: dict[str, torch.Tensor] = {}
        for path in fields:
            spec = BY_PATH[path]
            dtype = torch.bool if spec.dtype == "bool" else (
                torch.float64 if spec.dtype == "float64" else torch.float32)
            fill = False if dtype == torch.bool else float("nan")
            self._buf[path] = torch.full(
                (self.n_envs, expected_rows(spec, self.n_steps)) + spec.width, fill,
                dtype=dtype, device=self.device)
        self.last_inst = -1
        self.last_intv = -1

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self._buf)

    def put(self, path: str, k: int, value: torch.Tensor | np.ndarray | float) -> None:
        """Write row ``k`` for all envs; ``value`` broadcastable to (E, *width)."""
        buf = self._buf[path]
        buf[:, k] = torch.as_tensor(value, dtype=buf.dtype, device=self.device)
        if BY_PATH[path].sem == "inst":
            self.last_inst = max(self.last_inst, k)
        else:
            self.last_intv = max(self.last_intv, k)

    def truncate(self, n_steps: int) -> None:
        """Shrink to ``n_steps`` intervals (early termination)."""
        for path, buf in self._buf.items():
            self._buf[path] = buf[:, : expected_rows(BY_PATH[path], n_steps)]
        self.n_steps = int(n_steps)

    def to_numpy(self, env: int = 0) -> dict[str, np.ndarray]:
        return {p: b[env].detach().cpu().numpy().astype(BY_PATH[p].dtype)
                for p, b in self._buf.items()}
