"""Post-hoc split helpers (plan §6.5-DS8).

Split *roles* (train / val_id / test_id / *_ood_*) are assigned at plan time
to whole groups and stored in metadata; nothing here reassigns them. The only
post-hoc operation is the sample-efficiency study: nested, group-aware subsets
of the training pool (10/25/50/100 %, thesis RQ3).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def nested_group_subsets(group_ids: Sequence[str], fractions: Sequence[float],
                         seed: int) -> dict[float, set[str]]:
    """Nested subsets of *groups*: subset(f1) ⊂ subset(f2) for f1 < f2."""
    fr = sorted(float(f) for f in fractions)
    if any(not 0 < f <= 1 for f in fr):
        raise ValueError("fractions must be in (0, 1]")
    groups = np.array(sorted(set(group_ids)), dtype=object)
    order = np.random.default_rng(seed).permutation(len(groups))
    ranked = groups[order]
    return {f: set(ranked[: max(1, int(round(f * len(ranked))))].tolist()) for f in fr}
