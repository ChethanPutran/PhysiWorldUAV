#!/usr/bin/env python3
"""Job list of sweep `v7_stage1` (2026-11-26): the first training runs on pilot v7 (20 Hz). Seven runs, seed 0:
B2, B3, B4, B4 (B3 width), B5a, B5b and a small B3 (60k parameters).

    python scripts/sweep_jobs_v7_stage1.py > outputs/sweeps/v7_stage1/jobs.json
Run names st_v07_<model>[_<variant>]_s<seed>. Config: default.yaml + compare.yaml + v7_openloop.yaml (early-ended episodes used for TRAINING only,
statistics and loss weights fitted on closed-loop episodes); the small B3 adds config/model/b3_small.yaml."""
import json

DS = "data/datasets/physiworld_pilot_v7_v0.7.0"
BASE = ["config/training/default.yaml", "config/training/compare.yaml", "config/training/v7_openloop.yaml"]
jobs = []


def add(model, variant="", cfgs=(), seed=0):
    name = f"st_v07_{model}{('_' + variant) if variant else ''}_s{seed}"
    jobs.append({"name": name, "dataset": DS, "model": model, "configs": BASE + list(cfgs), "sets": [f"seed={seed}"]})


for m in ("b3", "b4", "b4_rw", "b2", "b5a", "b5b"):      # longest first
    add(m)
add("b3", "small", cfgs=["config/model/b3_small.yaml"])
print(json.dumps(jobs, indent=1))
