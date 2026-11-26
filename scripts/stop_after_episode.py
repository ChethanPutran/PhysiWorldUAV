#!/usr/bin/env python3
"""Stop generation workers right after they finish an episode (so that no work in progress is lost). Run on the HOST.

    python3 scripts/stop_after_episode.py data/datasets/physiworld_pilot_v7_v0.7.0 --experiment pilot_v7 --workers all
    python3 scripts/stop_after_episode.py ... --workers 0,3,6 [--dry-run]

For every selected worker it waits until the worker writes a new episode (its ``index.workerNN.json`` changes), then immediately sends SIGTERM,
and SIGKILL after --grace seconds if the Isaac process has not exited. At that moment the next episode has just started, so nothing half
written exists; ``--resume`` removes an incomplete episode anyway. It signals the processes through ``docker exec`` (they belong to root in the container, host pid -> container pid via NSpid). It stops only the python worker processes, not ``generate_parallel.sh``:
stop that script first (kill its PID from ``ps``), otherwise it starts the workers again after its attempt ends.
Keep the same worker count when you start the workers again (the episode split depends on it; see HANDOFF.md).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from physiworld_uav.visualization.dataset_report import worker_pids


def written(d: Path, w: int) -> int:
    try:
        return len(json.loads((d / f"index.worker{w:02d}.json").read_text()).get("episodes", []))
    except (OSError, json.JSONDecodeError):
        return -1


def alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()                    # the workers are root's (container) processes: os.kill(pid, 0) would be refused


def container_pid(pid: int) -> int:
    """The pid inside the container (last entry of NSpid in /proc/<pid>/status); the host pid differs."""
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("NSpid:"):
            return int(line.split()[-1])
    return pid


def signal_worker(container: str, pid: int, sig: str) -> None:
    subprocess.run(["docker", "exec", container, "kill", f"-{sig}", str(container_pid(pid))], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path); ap.add_argument("--experiment", required=True)
    ap.add_argument("--workers", default="all", help="all, or a comma list such as 0,3,6")
    ap.add_argument("--container", default="isaac-sim"); ap.add_argument("--grace", type=float, default=20.0); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pids = worker_pids(a.experiment)
    wanted = sorted(pids) if a.workers == "all" else [int(x) for x in a.workers.split(",")]
    todo = {w: pids[w] for w in wanted if w in pids}
    print(f"waiting for the next finished episode of workers {sorted(todo)} (pids {[todo[w] for w in sorted(todo)]})", flush=True)
    base = {w: written(a.dataset, w) for w in todo}
    if a.dry_run:
        print("dry run: nothing will be stopped"); return
    while todo:
        for w, pid in list(todo.items()):
            if not alive(pid):
                print(f"worker {w}: already gone", flush=True); del todo[w]; continue
            if written(a.dataset, w) > base[w]:
                print(f"worker {w}: episode finished ({written(a.dataset, w)} written); SIGTERM pid {pid}", flush=True)
                signal_worker(a.container, pid, "TERM"); t0 = time.time()
                while alive(pid) and time.time() - t0 < a.grace:
                    time.sleep(1)
                if alive(pid):
                    signal_worker(a.container, pid, "KILL"); print(f"worker {w}: SIGKILL", flush=True)
                del todo[w]
        time.sleep(0.5)
    print("done", flush=True)


if __name__ == "__main__":
    main()
