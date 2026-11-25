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

Integrity: every episode carries ``attrs["checksum"] = "sha256:<hex>"`` over its arrays (path, dtype, shape and bytes, in sorted
path order), written last. ``verify_shard`` recomputes it: a bit flip, a truncated dataset or a partial write is detected.
Episodes written before checksums existed have none (``missing``); ``add_missing_checksums`` adds them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping

import hashlib

import h5py
import numpy as np

from .metadata import EpisodeMetadata
from .schema import BY_PATH, validate_arrays


class SchemaError(ValueError):
    pass


def episode_checksum(arrays: Mapping[str, np.ndarray]) -> str:
    """SHA-256 over the episode arrays in sorted path order (path, dtype, shape, C-order bytes)."""
    h = hashlib.sha256()
    for path in sorted(arrays):
        a = np.ascontiguousarray(arrays[path])
        h.update(path.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


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
        g.attrs["checksum"] = episode_checksum(arrays)          # written last: an episode without it was interrupted
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


def verify_episode_group(g: h5py.Group) -> str:
    """"ok" | "mismatch" | "missing" (no checksum attribute) | "unreadable" (data cannot be read back)."""
    try:
        stored = g.attrs.get("checksum")
        if stored is None:
            return "missing"
        stored = stored.decode() if isinstance(stored, bytes) else str(stored)
        return "ok" if episode_checksum(_collect(g)) == stored else "mismatch"
    except Exception:  # noqa: BLE001  (truncated chunk, bad compression stream, …)
        return "unreadable"


def verify_shard(shard: str | Path) -> dict[str, str]:
    """episode_id → status for every episode of the shard (read-only)."""
    with h5py.File(shard, "r") as f:
        return {eid: verify_episode_group(f["episodes"][eid]) for eid in sorted(f["episodes"])}


def add_missing_checksums(shard: str | Path) -> int:
    """Write a checksum for every episode that has none (legacy shards). Returns how many were added.
    This only adds an attribute; the data arrays are untouched."""
    n = 0
    with h5py.File(shard, "a") as f:
        for eid in sorted(f["episodes"]):
            g = f["episodes"][eid]
            if "checksum" not in g.attrs and "metadata" in g.attrs:
                g.attrs["checksum"] = episode_checksum(_collect(g))
                n += 1
        f.flush()
    return n
