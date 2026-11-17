"""HDF5 episode storage (plan §6.5; format confirmed by the Run 5 benchmark, DS-STORE).

Layout of one shard file::

    shard_<worker>.h5
      /episodes/<episode_id>/time/t
                            /ground_truth/...
                            /observations/...
                            /actions/...
                            /control/...           (closed loop only)
                            /environment/...
                            attrs["metadata"] = EpisodeMetadata JSON

Every array is validated against ``data/schema.py`` before it is written; a
schema violation raises rather than producing a partially valid dataset.
Each write is flushed so an interrupted run leaves complete episodes only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping

import h5py
import numpy as np

from .metadata import EpisodeMetadata
from .schema import BY_PATH, validate_arrays


class SchemaError(ValueError):
    pass


class ShardWriter:
    def __init__(self, path: str | Path, *, compression: str | None = "gzip",
                 compression_opts: int | None = 4) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self.path, "a")
        self._f.require_group("episodes")
        self.compression = compression
        self.compression_opts = compression_opts if compression == "gzip" else None

    def write(self, arrays: Mapping[str, np.ndarray], meta: EpisodeMetadata) -> None:
        arrays = {k: np.asarray(v, dtype=BY_PATH[k].dtype) if k in BY_PATH else np.asarray(v)
                  for k, v in arrays.items()}
        issues = validate_arrays(arrays, data_type=meta.data_type)
        if issues:
            raise SchemaError(f"episode {meta.episode_id}: " + "; ".join(issues))
        eps = self._f["episodes"]
        if meta.episode_id in eps:
            raise SchemaError(f"episode {meta.episode_id} already exists in {self.path}")
        g = eps.create_group(meta.episode_id)
        for path, arr in arrays.items():
            chunked = arr.ndim >= 1 and arr.shape[0] > 1
            g.create_dataset(
                path, data=arr,
                compression=self.compression if chunked else None,
                compression_opts=self.compression_opts if chunked else None,
                chunks=(min(arr.shape[0], 1024),) + arr.shape[1:] if chunked else None,
            )
        g.attrs["metadata"] = meta.to_json()
        self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _collect(group: h5py.Group, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, item in group.items():
        path = f"{prefix}{name}"
        if isinstance(item, h5py.Dataset):
            out[path] = item[()]
        else:
            out.update(_collect(item, path + "/"))
    return out


def read_episode(shard: str | Path, episode_id: str) -> tuple[dict[str, np.ndarray], EpisodeMetadata]:
    with h5py.File(shard, "r") as f:
        g = f["episodes"][episode_id]
        return _collect(g), EpisodeMetadata.from_json(g.attrs["metadata"])


def iter_episodes(shard: str | Path) -> Iterator[tuple[dict[str, np.ndarray], EpisodeMetadata]]:
    with h5py.File(shard, "r") as f:
        for eid in sorted(f["episodes"]):
            g = f["episodes"][eid]
            yield _collect(g), EpisodeMetadata.from_json(g.attrs["metadata"])


def list_metadata(shard: str | Path) -> list[EpisodeMetadata]:
    with h5py.File(shard, "r") as f:
        return [EpisodeMetadata.from_json(f["episodes"][eid].attrs["metadata"])
                for eid in sorted(f["episodes"])]
