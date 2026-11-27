"""Figures of evaluation results (the JSON files written by ``physiworld_uav.evaluation.run``). Pure numpy/matplotlib.

    python -m physiworld_uav.visualization evaluation --json outputs/eval/st_v07_c0.json outputs/eval/st_v07_c1.json \\
        outputs/eval/st_v07_c2.json --out outputs/reports/evaluation_v7 [--rename st_v07_=] [--models b1 st_v07_b3_s0 ...]

Several JSON files are merged (model names must be distinct). Writes in ``--out``:
  error_vs_horizon.png  position RMSE against rollout length, one panel per role, one line per model (log axis)
  heatmap_position.png  models x roles, position RMSE at the last horizon (colour = log10; numbers are metres)
  heatmap_attitude.png  the same for the attitude error [deg]
  ood_gap.png           per model: RMSE on each OOD role divided by the RMSE on test_id (>1 = worse than in distribution)
  evaluation_tables.csv all numbers used"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

PAL = ["#000000", "#888888", "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7B4EA3", "#B8860B", "#444444", "#999933"]
ROLE_ORDER = ["val_id", "test_id", "test_ood_magnitude", "test_ood_direction", "test_ood_temporal", "test_ood_gust", "test_ood_physics",
              "test_ood_trajectory"]


def merge(paths, rename: str = "", models: list[str] | None = None) -> dict:
    out: dict = {}
    for p in paths:
        for m, per in json.loads(Path(p).read_text()).items():
            if m in out:
                raise SystemExit(f"model {m!r} appears in two files")
            out[m] = per
    if models:
        out = {m: out[m] for m in models if m in out}
    if rename:
        old, _, new = rename.partition("=")
        out = {m.replace(old, new): v for m, v in out.items()}
    return out


def roles_of(res: dict) -> list[str]:
    have = {r for per in res.values() for r, v in per.items() if v.get("n_windows")}
    return [r for r in ROLE_ORDER if r in have] + sorted(have - set(ROLE_ORDER) - {r for r in have if r.startswith("physics_validation")})


def last_k(r: dict, stem: str) -> str:
    return max(r[stem], key=int)


def heatmap(res: dict, roles: list[str], stem: str, unit: str, path: Path, title: str) -> None:
    models = list(res)
    M = np.full((len(models), len(roles)), np.nan)
    for i, m in enumerate(models):
        for j, r in enumerate(roles):
            v = res[m].get(r, {})
            if v.get("n_windows"):
                M[i, j] = v[stem][last_k(v, stem)]
    fig, ax = plt.subplots(figsize=(1.3 * len(roles) + 3, 0.5 * len(models) + 2.2))
    im = ax.imshow(M, aspect="auto", cmap="viridis_r", norm=LogNorm(vmin=np.nanmin(M[M > 0]), vmax=np.nanmax(M)))
    ax.set_xticks(range(len(roles)), [r.replace("test_ood_", "ood ").replace("_", " ") for r in roles], rotation=30, ha="right")
    ax.set_yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(roles)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3g}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > np.sqrt(np.nanmin(M[M > 0]) * np.nanmax(M)) else "black")
    fig.colorbar(im, label=f"{unit} (log scale)")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--rename", default="", help="OLD=NEW substring replaced in the model names (shorter legends)")
    a = ap.parse_args(argv)
    res = merge(a.json, a.rename, a.models)
    if not res:
        raise SystemExit("no models")
    a.out.mkdir(parents=True, exist_ok=True)
    roles = roles_of(res)
    colors = {m: PAL[i % len(PAL)] for i, m in enumerate(res)}

    n = len(roles); cols = min(4, n); rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.6 * rows), squeeze=False, sharey=True)
    for ax, role in zip(axs.ravel(), roles):
        for m in res:
            v = res[m].get(role, {})
            if v.get("n_windows"):
                ks = sorted(v["pos_rmse_m"], key=int)
                ax.plot([int(k) for k in ks], [v["pos_rmse_m"][k] for k in ks], "-o", ms=3, color=colors[m], label=m)
        ax.set(title=f"{role} ({next(res[m][role]['n_episodes'] for m in res if res[m].get(role, {}).get('n_windows'))} ep)",
               xlabel="rollout length k [steps]", yscale="log"); ax.grid(alpha=0.3, which="both")
    for ax in axs[:, 0]:
        ax.set_ylabel("position RMSE [m]")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 6), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(a.out / "error_vs_horizon.png", dpi=110); plt.close(fig)

    heatmap(res, roles, "pos_rmse_m", "m", a.out / "heatmap_position.png", "Position RMSE at the longest horizon [m]")
    heatmap(res, roles, "att_rmse_deg", "deg", a.out / "heatmap_attitude.png", "Attitude RMSE at the longest horizon [deg]")

    oods = [r for r in roles if "ood" in r]
    if "test_id" in roles and oods:
        fig, ax = plt.subplots(figsize=(1.6 * len(oods) * len(res) / 4 + 5, 4))
        w = 0.8 / len(res)
        for i, m in enumerate(res):
            base = res[m].get("test_id", {})
            if not base.get("n_windows"):
                continue
            b = base["pos_rmse_m"][last_k(base, "pos_rmse_m")]
            vals = [res[m][r]["pos_rmse_m"][last_k(res[m][r], "pos_rmse_m")] / b if res[m].get(r, {}).get("n_windows") else np.nan for r in oods]
            ax.bar(np.arange(len(oods)) + i * w, vals, w, color=colors[m], label=m)
        ax.axhline(1, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(oods)) + 0.4 - w / 2, [r.replace("test_ood_", "") for r in oods])
        ax.set(ylabel="RMSE(OOD role) / RMSE(test_id), last k", yscale="log"); ax.grid(alpha=0.3, axis="y", which="both"); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(a.out / "ood_gap.png", dpi=110); plt.close(fig)

    with open(a.out / "evaluation_tables.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["model", "role", "episodes", "windows", "k", "pos_rmse_m", "att_rmse_deg", "vel_rmse_mps", "reliable_horizon"])
        for m in res:
            for r in roles:
                v = res[m].get(r, {})
                for k in sorted(v.get("pos_rmse_m", {}), key=int):
                    wr.writerow([m, r, v["n_episodes"], v["n_windows"], k, v["pos_rmse_m"][k], v["att_rmse_deg"][k], v["vel_rmse_mps"][k],
                                 v["reliable_horizon_mean"]])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
