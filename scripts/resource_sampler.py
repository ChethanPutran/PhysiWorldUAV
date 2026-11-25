#!/usr/bin/env python3
"""Sample the resource use of running training processes (host side, stdlib only).

    python3 scripts/resource_sampler.py --out outputs/reports/training_compare_v05_v06 [--interval 2]

Every ``interval`` seconds, for every process whose command line contains ``physiworld_uav.training.run``:
  CPU % (from /proc/<pid>/stat deltas), resident memory and its kernel-tracked peak (VmRSS, VmHWM), GPU memory used by the process
  (nvidia-smi --query-compute-apps), the GPU it runs on, and its current epoch / curriculum horizon (last line of
  outputs/experiments/<run>/metrics.jsonl). Also one row per GPU (utilisation, memory) per sample.
Writes ``resource_samples.csv`` and ``gpu_samples.csv`` into --out; stops 60 s after the last training process has ended.
Run it on the HOST (it reads /proc of the container's processes and calls the host's nvidia-smi).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

TICKS = os.sysconf("SC_CLK_TCK")


def training_processes() -> dict[int, dict]:
    out = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="replace") for a in args]
        if not any("physiworld_uav.training.run" in a for a in args):
            continue
        if args and args[0].endswith(("bash", "sh")):            # the wrapper shell, not the python process
            continue
        d = dict(zip(args, args[1:]))
        out[int(p.name)] = {"run": d.get("--name", "?"), "device": d.get("--device", "?"), "dataset": Path(d.get("--dataset", "?")).name,
                            "model": d.get("--model", "?")}
    return out


def cpu_ticks(pid: int) -> float | None:
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return float(f[11]) + float(f[12])                        # utime + stime
    except (OSError, IndexError):
        return None


def mem_mb(pid: int) -> tuple[float, float]:
    rss = hwm = 0.0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS"):
                rss = float(line.split()[1]) / 1024
            elif line.startswith("VmHWM"):
                hwm = float(line.split()[1]) / 1024
    except OSError:
        pass
    return rss, hwm


def smi(query: str, fields: str) -> list[list[str]]:
    try:
        r = subprocess.run(["nvidia-smi", f"--query-{query}={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return [[x.strip() for x in line.split(",")] for line in r.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def progress(run: str, exp_root: Path) -> tuple[str, str]:
    mp = exp_root / run / "metrics.jsonl"
    try:
        with mp.open("rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 4096))
            last = [l for l in f.read().decode(errors="replace").splitlines() if l.strip()][-1]
        d = json.loads(last)
        return str(d["epoch"]), str(d["horizon"])
    except Exception:
        return "", ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--exp-root", type=Path, default=Path("outputs/experiments"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fr, fg = (a.out / "resource_samples.csv").open("a", newline=""), (a.out / "gpu_samples.csv").open("a", newline="")
    wr, wg = csv.writer(fr), csv.writer(fg)
    if fr.tell() == 0:
        wr.writerow(["t_unix", "run", "dataset", "model", "pid", "gpu", "cpu_pct", "rss_mb", "rss_peak_mb", "gpu_mem_mb", "epoch", "horizon"])
        wg.writerow(["t_unix", "gpu", "util_pct", "mem_used_mb"])
    prev: dict[int, tuple[float, float]] = {}
    idle_since = None
    while True:
        t = time.time()
        procs = training_processes()
        gpu_mem = {int(r[0]): float(r[1]) for r in smi("compute-apps", "pid,used_memory") if r[0].isdigit()}
        for r in smi("gpu", "index,utilization.gpu,memory.used"):
            wg.writerow([f"{t:.1f}", *r])
        for pid, info in procs.items():
            ticks = cpu_ticks(pid)
            cpu = ""
            if ticks is not None and pid in prev and t > prev[pid][1]:
                cpu = f"{100.0 * (ticks - prev[pid][0]) / TICKS / (t - prev[pid][1]):.1f}"
            if ticks is not None:
                prev[pid] = (ticks, t)
            rss, hwm = mem_mb(pid)
            ep, hz = progress(info["run"], a.exp_root)
            gpu = info["device"].split(":")[-1]
            wr.writerow([f"{t:.1f}", info["run"], info["dataset"], info["model"], pid, gpu, cpu, f"{rss:.0f}", f"{hwm:.0f}", f"{gpu_mem.get(pid, 0):.0f}", ep, hz])
        fr.flush(); fg.flush()
        if procs:
            idle_since = None
        else:
            idle_since = idle_since or t
            if t - idle_since > 60:
                break
        time.sleep(max(0.0, a.interval - (time.time() - t)))


if __name__ == "__main__":
    main()
