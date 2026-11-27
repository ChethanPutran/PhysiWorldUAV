#!/usr/bin/env python3
"""Compare the position tracking error of two datasets generated from the SAME plan (same episode ids), e.g. the 50 Hz and the 20 Hz controller
of config/experiment/stability_20hz.yaml. Run INSIDE the container (needs h5py).

    python scripts/compare_tracking.py data/datasets/stab50_v0.1.0 data/datasets/stab20_v0.1.0 [--md out.md]

Per episode: RMS and maximum of the position tracking error norm ‖p − p_ref‖ after the first second (physics-rate samples), and the final
status. Table per scenario and part (calm / windy)."""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def load(d: Path) -> dict:
    meta = {json.loads(l)["episode_id"]: json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip()}
    out = {}
    for f in sorted(glob.glob(str(d / "shard_*.h5"))):
        with h5py.File(f) as h:
            for eid, g in h["episodes"].items():
                if "control/tracking_error/position_W" not in g:
                    continue
                e = np.linalg.norm(g["control/tracking_error/position_W"][()], axis=1)[200:]
                out[eid] = {"rms": float(np.sqrt((e ** 2).mean())) if len(e) else float("nan"), "max": float(e.max()) if len(e) else float("nan"),
                            "scenario": meta[eid]["scenario_template"], "part": meta[eid]["part"], "n": len(e)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path); ap.add_argument("b", type=Path); ap.add_argument("--md", type=Path)
    x = ap.parse_args()
    A, B = load(x.a), load(x.b)
    common = sorted(set(A) & set(B))
    rows = defaultdict(list)
    for e in common:
        rows[(A[e]["part"], A[e]["scenario"])].append((A[e]["rms"], B[e]["rms"], A[e]["max"], B[e]["max"]))
    lines = [f"Position tracking error after 1 s, {x.a.name} (A) vs {x.b.name} (B), {len(common)} common episodes ({len(A)} / {len(B)} found)", "",
             "| Part | Scenario | n | RMS A [m] | RMS B [m] | RMS B / A | max A [m] | max B [m] |", "|---|---|---|---|---|---|---|---|"]
    for (part, sc), v in sorted(rows.items()):
        v = np.array(v)
        lines.append(f"| {part} | {sc} | {len(v)} | {v[:, 0].mean():.3f} | {v[:, 1].mean():.3f} | {v[:, 1].mean() / max(v[:, 0].mean(), 1e-9):.2f} | {v[:, 2].max():.3f} | {v[:, 3].max():.3f} |")
    allv = np.array([r for v in rows.values() for r in v])
    if len(allv):
        lines.append(f"| all | all | {len(allv)} | {allv[:, 0].mean():.3f} | {allv[:, 1].mean():.3f} | {allv[:, 1].mean() / max(allv[:, 0].mean(), 1e-9):.2f} | {allv[:, 2].max():.3f} | {allv[:, 3].max():.3f} |")
    print("\n".join(lines))
    if x.md:
        x.md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
