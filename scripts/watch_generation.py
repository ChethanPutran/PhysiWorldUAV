#!/usr/bin/env python3
"""Print an event line when something changes in a parallel generation (for a monitor): worker count changes, a worker log with a fatal
error, progress every 20 minutes, completion. Quiet otherwise.

    python3 scripts/watch_generation.py data/datasets/physiworld_pilot_v7_v0.7.0 --experiment pilot_v7 --workers 9
"""
from __future__ import annotations

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))

import argparse
import glob
import json
import subprocess
import time
from pathlib import Path

from physiworld_uav.notify import notify


def alive(exp: str) -> int:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return sum(1 for l in out.splitlines() if "generate run" in l and "--shard" in l and f"experiment={exp}" in l and not l.lstrip().startswith(("bash", "/bin/bash")))


def written(d: Path) -> int:
    n = 0
    for f in glob.glob(str(d / "index.worker*.json")):
        try:
            n += len(json.loads(Path(f).read_text()).get("episodes", []))
        except (OSError, json.JSONDecodeError):
            pass
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path); ap.add_argument("--experiment", required=True); ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--total", type=int, default=0); ap.add_argument("--progress-min", type=float, default=20)
    a = ap.parse_args()
    total = a.total or sum(1 for _ in (a.dataset / "manifest.jsonl").open())
    seen_fatal: set[str] = set(); last_alive = None; t_prog = 0.0; t0 = time.time(); n_first = None
    while True:
        for f in glob.glob("logs/gen_2*_w*.log"):
            if f not in seen_fatal and "Fatal Python error" in Path(f).read_text(errors="replace"):
                seen_fatal.add(f); print(f"WORKER CRASH (fatal error): {f}", flush=True)
                notify("A generation worker crashed (fatal error in its log)", title="Data generation", priority="high", tags="rotating_light")
        n_alive = alive(a.experiment)
        if n_alive != last_alive:
            print(f"workers running: {n_alive} of {a.workers}", flush=True); last_alive = n_alive
        n = written(a.dataset)
        if n_first is None and n > 0:
            n_first, t_first = n, time.time()
        if time.time() - t_prog > a.progress_min * 60:
            t_prog = time.time()
            rate = (n - n_first) / (time.time() - t_first) * 3600 if n_first is not None and time.time() - t_first > 120 and n > n_first else None
            print(f"{time.strftime('%H:%M')} {n}/{total} episodes ({100 * n / total:.1f} %)" + (f", {rate:.0f} episodes/h, ETA {(total - n) / rate:.1f} h" if rate else ", rate n/a yet"), flush=True)
        idx = a.dataset / "index.json"
        if idx.exists():
            try:
                if json.loads(idx.read_text()).get("complete"):
                    print("GENERATION COMPLETE", flush=True); return
            except json.JSONDecodeError:
                pass
        if n_alive == 0 and n > 0 and time.time() - t0 > 300 and not idx.exists():
            print(f"no worker running and no merged index yet ({n}/{total} written)", flush=True)
            notify(f"No generation worker is running and the dataset is not complete ({n}/{total} episodes)", title="Data generation", priority="high", tags="warning")
            time.sleep(300)
        time.sleep(30)


if __name__ == "__main__":
    main()
