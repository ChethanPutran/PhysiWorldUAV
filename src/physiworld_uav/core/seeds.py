"""Deterministic, hierarchical seed derivation (plan §6.6-RP1).

Every random stream is derived from one experiment seed through a tree of
*stable string keys*:

    global seed
      └ "split:train" └ "domain:dom_00003" └ "scenario:scn_00012"
          └ "episode:..." └ stream "wind" / "trajectory" / "excitation" / ...

A child seed depends only on its parent's seed and its own key, never on the
order in which siblings are created. Adding a scenario type, a new stream or
an extra initial-condition field therefore never shifts unrelated seeds.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# Named per-episode streams (plan §6.6; design doc §11.1 independent streams).
STREAMS: tuple[str, ...] = (
    "initial_conditions",
    "trajectory",
    "wind",
    "excitation",
    "physics_noise",
    "sensor_noise",
    "domain",
)


def _key_words(key: str) -> tuple[int, ...]:
    """Stable 128-bit hash of ``key`` as four uint32 words (no Python ``hash``)."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()[:16]
    return tuple(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4))


@dataclass(frozen=True)
class SeedNode:
    """A node in the seed tree. Immutable; children are derived on demand."""

    entropy: int
    path: tuple[str, ...] = ()

    def child(self, key: str) -> "SeedNode":
        if not key:
            raise ValueError("seed keys must be non-empty strings")
        return SeedNode(entropy=self.entropy, path=self.path + (key,))

    def descend(self, *keys: str) -> "SeedNode":
        node = self
        for k in keys:
            node = node.child(k)
        return node

    @property
    def sequence(self) -> np.random.SeedSequence:
        spawn_key: tuple[int, ...] = ()
        for k in self.path:
            spawn_key += _key_words(k)
        return np.random.SeedSequence(entropy=self.entropy, spawn_key=spawn_key)

    @property
    def seed(self) -> int:
        """A 31-bit integer seed (for APIs that need an int, e.g. torch)."""
        return int(self.sequence.generate_state(1, dtype=np.uint32)[0] >> 1)

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.sequence)

    @property
    def path_str(self) -> str:
        return "/".join(self.path)


def root(seed: int) -> SeedNode:
    if seed < 0:
        raise ValueError("global seed must be non-negative")
    return SeedNode(entropy=int(seed))


def child_seed(seed: int, key: str) -> int:
    """Derive an int seed for ``key`` from an int ``seed`` (used inside components)."""
    return root(seed).child(key).seed
