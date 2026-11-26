#!/usr/bin/env python3
"""Progress and ETA of a parallel generation (reads the per-worker index files of the dataset directory). Run on the host or in the container.

    python3 scripts/gen_progress.py data/datasets/physiworld_pilot_v7_v0.7.0 [--total 4530] [--watch 60]
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path


def snapshot(d: Path, total: int) -> tuple[int, dict]:
    per, n = {}, 0
    for f in sorted(glob.glob(str(d / "index.worker*.json"))):
        try:
            w = json.loads(Path(f).read_text()); per[Path(f).stem.split(".")[-1]] = w.get("n_written", 0); n += w.get("n_written", 0)
        except (OSError, json.JSONDecodeError):
            pass                                                   # a worker is rewriting its index
    return n, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path); ap.add_argument("--total", type=int, default=0); ap.add_argument("--watch", type=float, default=0)
    a = ap.parse_args()
    total = a.total or sum(1 for _ in (a.dataset / "manifest.jsonl").open())
    t0, n0 = time.time(), None
    while True:
        n, per = snapshot(a.dataset, total)
        if n0 is None:
            n0 = n
        el = time.time() - t0
        rate = (n - n0) / el * 3600 if el > 30 and n > n0 else None                  # episodes per hour since this view started
        eta = f"{(total - n) / rate:.1f} h" if rate else "n/a (needs 30 s of observation)"
        print(f"{time.strftime('%H:%M:%S')}  {n}/{total} episodes ({100 * n / total:.1f} %)  workers: " + " ".join(f"{k[-2:]}:{v}" for k, v in per.items())
              + f"  rate {rate:.0f}/h  ETA {eta}" if rate else f"{time.strftime('%H:%M:%S')}  {n}/{total} episodes ({100 * n / total:.1f} %)  workers: " + " ".join(f"{k[-2:]}:{v}" for k, v in per.items()) + f"  ETA {eta}", flush=True)
        if not a.watch or n >= total:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
