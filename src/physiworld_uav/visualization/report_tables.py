"""LaTeX tables of the report that are computed from the project's files (reusable for later reports).

    python -m physiworld_uav.visualization tables [--out outputs/tables.tex]

Writes ``\\newcommand`` macros: \\TabMainResults (ID / OOD position and velocity RMSE), \\TabDatasetStats (V0.5, V0.6, V7), \\TabTrainConfigs
(Smoke, V0.5, V0.6, V7) and \\TabVersions (dataset versions with size), each the body rows of a tabular.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OOD6 = ["test_ood_magnitude", "test_ood_direction", "test_ood_temporal", "test_ood_gust", "test_ood_physics", "test_ood_trajectory"]


def _ev(*files):
    E = {}
    for f in files:
        for m, r in json.load(open(ROOT / "outputs/eval" / f)).items():
            E[m] = r
    return E


def main_results() -> str:
    E = _ev("kv_v07_c0.json", "kv_v07_c1.json", "kv_v07_c2.json", "kv_v07_d_c1.json", "kv_v07_d_c2.json"); S = _ev("st_v07_c0.json")
    err = json.load(open(ROOT / "outputs/figure_cache/err_data.json")) if (ROOT / "outputs/figure_cache/err_data.json").exists() else []
    rows = [("Persistence (B0)", S["b0"]), ("Constant velocity", None), ("Physics only (B1)", E["b1_kv"]), ("Learned dynamics (B5b)", E["kv_v07_b5b_s0"]),
            ("Latent baseline (B2)", E["kv_v07_b2_s0"]), ("No memory (B4, B3 width)", E["kv_v07_b4_rw_s0"]), ("Proposed (B3)", E["kv_v07_b3_s0"])]
    table = []
    for name, m in rows:
        if m is None:           # constant velocity from the per-episode errors (windows every 10 steps)
            def agg(roles, key):
                v = [np.mean([r[key] ** 2 for r in err if r["model"] == "cv" and r["role"] == ro]) ** .5 for ro in roles]; return float(np.mean(v)) if v else float("nan")
            vals = (agg(["test_id"], "pos_rmse_20"), agg(OOD6, "pos_rmse_20"), agg(["test_id"], "vel_rmse_20"), agg(OOD6, "vel_rmse_20"))
        else:
            g = lambda role, k: m[role][k]["20"]   # noqa: E731
            vals = (g("test_id", "pos_rmse_m"), float(np.mean([g(r, "pos_rmse_m") for r in OOD6])), g("test_id", "vel_rmse_mps"), float(np.mean([g(r, "vel_rmse_mps") for r in OOD6])))
        table.append((name, vals))
    best = [min(v[k] for _, v in table if not np.isnan(v[k])) for k in range(4)]      # bold the lowest value of each column
    out = []
    for name, vals in table:
        cells = " & ".join(f"\\textbf{{{v:.3f}}}" if abs(v - best[k]) < 5e-4 else f"{v:.3f}" for k, v in enumerate(vals))
        out.append(f"{name} & {cells}\\\\")
    return "\n".join(out)


def versions() -> list[dict]:
    spec = [("smoke_v0.3.0", "Smoke v0.3", 4), ("physiworld_pilot_v0.4.0-dev", "v0.4-dev", 4), ("physiworld_pilot_v0.5.0", "v0.5", 4), ("physiworld_pilot_v0.6.0", "v0.6", 4), ("physiworld_pilot_v7_v0.7.0", "v7", 10)]
    out = []
    for d, lab, dec in spec:
        p = ROOT / "data/datasets" / d; j = json.load(open(p / "index.json")); eps = j["episodes"]; n = collections.Counter()
        for e in eps:
            r = e["split_role"]; k = "train" if r == "train" else "val" if r.startswith("val") else "idtest" if r == "test_id" else "oodtest" if r.startswith("test_ood") else "phys"
            n[k] += e["n_steps"] // dec
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
        out.append(dict(lab=lab, n=len(eps), steps=sum(n.values()), hours=sum(e["n_steps"] for e in eps) * 0.005 / 3600, roles=n, wall=j.get("generation_wall_s"), mb=size, dec=dec, n_ep=collections.Counter(e["split_role"] for e in eps)))
    return out


def dataset_stats() -> str:
    V = {v["lab"]: v for v in versions()}; cols = ["v0.5", "v0.6", "v7"]
    def row(name, f): return name + " & " + " & ".join(f(V[c]) for c in cols) + "\\\\"
    gt = lambda v: f"{v['wall'] / 3600:.2f} h" if (v["wall"] and v["wall"] > 120) else "n/a (resumed)"   # noqa: E731
    return "\n".join([row("Trajectories", lambda v: f"{v['n']:,}"), row("Total model steps (samples)", lambda v: f"{v['steps']:,}"), row("Simulated duration", lambda v: f"{v['hours']:.2f} h"),
                      row("Mean episode length", lambda v: f"{v['hours'] * 3600 / v['n']:.1f} s"), row("Train steps", lambda v: f"{v['roles']['train']:,}"), row("Validation steps", lambda v: f"{v['roles']['val']:,}"),
                      row("ID test steps", lambda v: f"{v['roles']['idtest']:,}"), row("OOD test steps", lambda v: f"{v['roles']['oodtest']:,}"), row("Generation time", gt), row("Disk usage", lambda v: f"{v['mb']:.0f} MB")])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0]); ap.add_argument("--out", type=Path, default=ROOT / "outputs/tables.tex"); a = ap.parse_args(argv)
    txt = "\n".join(f"\\newcommand{{\\{n}}}{{%\n{f()}%\n}}" for n, f in (("TabMainResults", main_results), ("TabDatasetStats", dataset_stats)))
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(txt); print("wrote", a.out)


if __name__ == "__main__":
    main()
