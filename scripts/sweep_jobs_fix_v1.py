#!/usr/bin/env python3
"""Job list of sweep `fix_v1` (2026-11-25): repeat the runs affected by the best.pt bug, seeds, B3a-c, learning rate / weight decay.

    python scripts/sweep_jobs_fix_v1.py > outputs/sweeps/fix_v1/jobs.json
Names: fx_<dataset>_<model>[_<variant>]_s<seed>. All with config/training/default.yaml + compare.yaml (+ model YAML / overrides).
"""
import json

D = {"v05": "data/datasets/physiworld_pilot_v0.5.0", "v06": "data/datasets/physiworld_pilot_v0.6.0"}
BASE = ["config/training/default.yaml", "config/training/compare.yaml"]
jobs = []


def add(ds, model, seed=0, variant="", cfgs=(), sets=()):
    name = f"fx_{ds}_{model}{('_' + variant) if variant else ''}_s{seed}"
    jobs.append({"name": name, "dataset": D[ds], "model": model, "configs": BASE + list(cfgs), "sets": [f"seed={seed}", *sets],
                 "group": variant or "rerun" if seed == 0 else "seeds"})


# 1. the eight runs affected by the best.pt bug, seed 0, fixed trainer
for m in ("b4", "b4_rw", "b5b"):
    add("v05", m)
for m in ("b2", "b4", "b4_rw", "b5a", "b5b"):
    add("v06", m)
# 1b. seeds 1 and 2 for B3, B4, B5b on v0.6 (B3 seed 0 is the existing v06_b3: not affected by the bug)
for s in (1, 2):
    for m in ("b3", "b4", "b5b"):
        add("v06", m, seed=s)
# 2. spec 3.4 variants of B3 (v0.6, seeds 0-2); B3 seed 0 = existing v06_b3
for v in ("a", "b", "c"):
    for s in (0, 1, 2):
        add("v06", "b3", seed=s, variant="b3" + v, cfgs=[f"config/model/b3{v}.yaml"])
# 3. lower learning rate / more weight decay for B4 and B5b (v0.6, seed 0)
for m in ("b4", "b5b"):
    add("v06", m, variant="lr2e-4", sets=["lr=0.0002"])
    add("v06", m, variant="wd1e-2", sets=["weight_decay=0.01"])
    add("v06", m, variant="lr2e-4_wd1e-2", sets=["lr=0.0002", "weight_decay=0.01"])
for j in jobs:
    j.pop("group", None)
order = {"b3": 0, "b2": 1, "b5a": 1, "b4": 2, "b4_rw": 2, "b5b": 3}      # longest first
jobs.sort(key=lambda j: order.get(j["model"], 9))
print(json.dumps(jobs, indent=1))
