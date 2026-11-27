#!/usr/bin/env python3
"""Job list of sweep `v7_kv` (2026-11-27): the stage-1 models retrained with the TRUE vehicle parameters of each episode in the physics prior
(`known_vehicle`) and given to B2 / B5 as an input (`vehicle_input`); plus B6b (B3 + oracle wrench), future and current.
Nine runs, seed 0: B2, B3, B4, B4 (B3 width), B5a, B5b, B3 small (60k), B6b future, B6b current.

    python scripts/sweep_jobs_v7_kv.py > outputs/sweeps/v7_kv/jobs.json
Run names kv_v07_<model>[_<variant>]_s<seed>. Config: default + compare + v7_openloop + known_vehicle (the small B3 adds config/model/b3_small.yaml)."""
import json

DS = "data/datasets/physiworld_pilot_v7_v0.7.0"
BASE = ["config/training/default.yaml", "config/training/compare.yaml", "config/training/v7_openloop.yaml", "config/training/known_vehicle.yaml"]
jobs = []


def add(model, variant="", cfgs=(), seed=0):
    name = f"kv_v07_{model}{('_' + variant) if variant else ''}_s{seed}"
    jobs.append({"name": name, "dataset": DS, "model": model, "configs": BASE + list(cfgs), "sets": [f"seed={seed}"]})


for m in ("b3", "b6b_future", "b6b_current", "b4", "b4_rw", "b2", "b5a", "b5b"):      # longest first
    add(m)
add("b3", "small", cfgs=["config/model/b3_small.yaml"])
print(json.dumps(jobs, indent=1))
