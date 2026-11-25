#!/usr/bin/env python3
"""Run a list of training jobs on several GPUs (several jobs per GPU), resumable. Run INSIDE the container.

    python scripts/run_sweep.py --jobs outputs/sweeps/<tag>/jobs.json --gpus 0,1,2 --per-gpu 3

jobs.json: a list of {"name", "dataset", "model", "configs": [yaml...], "sets": ["key=value"...]}. A job whose
outputs/experiments/<name>/summary.json already exists is skipped (so the script can be restarted). Output of each job:
logs/train_<name>.log; the state of all jobs is kept in <jobs dir>/status.json (queued / running / done / failed, GPU, times),
which scripts/live_monitor.py reads.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--per-gpu", type=int, default=3)
    ap.add_argument("--exp-root", type=Path, default=Path("outputs/experiments"))
    a = ap.parse_args()
    jobs = json.loads(a.jobs.read_text())
    status_path = a.jobs.parent / "status.json"
    lock = threading.Lock()
    st = {j["name"]: {"state": "queued", "model": j["model"], "dataset": Path(j["dataset"]).name} for j in jobs}
    queue = []
    for j in jobs:
        if (a.exp_root / j["name"] / "summary.json").exists():
            st[j["name"]]["state"] = "done"; st[j["name"]]["note"] = "already finished (skipped)"
        else:
            queue.append(j)

    def save() -> None:
        tmp = status_path.with_suffix(".tmp"); tmp.write_text(json.dumps(st, indent=1)); tmp.replace(status_path)

    save()
    env = {**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
           "MKL_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
    Path("logs").mkdir(exist_ok=True)

    def worker(gpu: str) -> None:
        while True:
            with lock:
                if not queue:
                    return
                j = queue.pop(0)
                st[j["name"]].update(state="running", gpu=gpu, started=time.strftime("%Y-%m-%d %H:%M:%S"), t0=time.time())
                save()
            cmd = [sys.executable, "-m", "physiworld_uav.training.run", "--dataset", j["dataset"], "--name", j["name"], "--model", j["model"],
                   "--device", f"cuda:{gpu}", "--config", *j["configs"]]
            if j.get("sets"):
                cmd += ["--set", *j["sets"]]
            with open(f"logs/train_{j['name']}.log", "w") as lg:
                rc = subprocess.call(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env)
            with lock:
                st[j["name"]].update(state="done" if rc == 0 else "failed", rc=rc, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                                     wall_s=time.time() - st[j["name"]]["t0"])
                save()

    ths = [threading.Thread(target=worker, args=(g,)) for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    for t in ths:
        t.start(); time.sleep(2)                         # stagger the starts (dataset loading)
    for t in ths:
        t.join()
    print("all jobs finished:", {k: sum(v["state"] == k for v in st.values()) for k in ("done", "failed")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
