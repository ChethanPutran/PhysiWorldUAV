"""Training / evaluation report: figures, tables and a self-contained HTML + Markdown report for a set of runs.

    python -m physiworld_uav.evaluation.report --tags v05 v06 --out outputs/reports/training_compare_v05_v06 \\
        [--eval outputs/eval] [--resources outputs/reports/training_compare_v05_v06]

Reads, per tag and model kind, ``outputs/experiments/<tag>_<kind>/`` (``metrics.jsonl``, ``summary.json``, ``provenance.json``,
``config.yaml``), the resource samples written by ``scripts/resource_sampler.py`` (``resource_samples.csv``, ``gpu_samples.csv``)
and the evaluation JSON files of ``evaluation.run`` (``<eval>/<tag>.json`` and ``<eval>/<tag>_on_<other>.json`` for cross-evaluation).
Writes ``figures/*.png``, ``tables/*.csv``, ``REPORT.md`` and ``report.html`` (images embedded). Anything missing is skipped and
said so in the report. Colour follows the entity (first tag blue, second orange) and uses the validated reference palette.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

KINDS = ["b2", "b3", "b4", "b4_rw", "b5a", "b5b", "b6b_future", "b6b_current"]
FIXED = ["b0", "b1", "b6a_future", "b6a_current"]
LABEL = {"b0": "B0 hold last", "b1": "B1 physics only", "b2": "B2 pure neural", "b3": "B3 physics+residual+memory", "b4": "B4 no memory",
         "b4_rw": "B4 (B3 width)", "b5a": "B5a learned field", "b5b": "B5b learned dynamics", "b6a_future": "B6a oracle (future)",
         "b6a_current": "B6a oracle (current)", "b6b_future": "B6b oracle (future)", "b6b_current": "B6b oracle (current)"}
USES_IMU = {"b2", "b3", "b5a", "b5b", "b6b_future", "b6b_current"}            # models whose latent reads the IMU (φ matters)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834"]                                               # categorical slots 1, 2 (validated: all checks pass)
DATASET_NOTE = {"v05": "v0.5.0 (raw Isaac IMU)", "v06": "v0.6.0 (corrected IMU)"}


# ------------------------------------------------------------------------------------------------ loading
def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_runs(exp_root: Path, tags: Sequence[str]) -> dict[tuple[str, str], dict]:
    runs = {}
    for tag in tags:
        for kind in KINDS:
            d = exp_root / f"{tag}_{kind}"
            if not d.exists():
                continue
            mp = d / "metrics.jsonl"
            rec = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()] if mp.exists() else []
            runs[(tag, kind)] = {"dir": d, "metrics": rec, "summary": _read_json(d / "summary.json"), "prov": _read_json(d / "provenance.json") or {},
                                 "config": _read_json(d / "config.yaml") or {}}
    return runs


def load_samples(res_dir: Path | None) -> tuple[list[dict], list[dict]]:
    out = []
    for name in ("resource_samples.csv", "gpu_samples.csv"):
        p = (res_dir / name) if res_dir else None
        out.append(list(csv.DictReader(p.open())) if p and p.exists() else [])
    return out[0], out[1]


# ------------------------------------------------------------------------------------------------ tables
def _val_key(v: Mapping[str, Any]) -> str | None:
    ks = [int(k) for k in (v or {}).keys()]
    return str(max(ks)) if ks else None


def runs_table(runs: dict, samples: list[dict]) -> list[dict]:
    by_run = defaultdict(list)
    for s in samples:
        by_run[s["run"]].append(s)
    rows = []
    for (tag, kind), r in runs.items():
        m, sm, pv = r["metrics"], r["summary"], r["prov"]
        vals = [(e["epoch"], e["val"]["pos_rmse_m"][_val_key(e["val"]["pos_rmse_m"])]) for e in m if e.get("val")]
        best = min(vals, key=lambda x: x[1]) if vals else (None, None)
        res = by_run.get(f"{tag}_{kind}", [])
        cpu = [float(s["cpu_pct"]) for s in res if s["cpu_pct"] not in ("", None)]
        gmem = [float(s["gpu_mem_mb"]) for s in res]
        rss_pk = [float(s["rss_peak_mb"]) for s in res]
        t_res = [float(s["t_unix"]) for s in res]
        ep_res = [e["resources"] for e in m if e.get("resources")]
        wall = (sm or {}).get("total_wall_s")
        row = dict(tag=tag, dataset=DATASET_NOTE.get(tag, tag), kind=kind, model=LABEL.get(kind, kind), status="finished" if sm else "running",
                   params=pv.get("n_params"), epochs=len(m), final_horizon=m[-1]["horizon"] if m else None,
                   stop_reason=(sm or {}).get("stop_reason", ""), best_epoch=best[0], best_val_pos_rmse_m=best[1],
                   val_horizon=int(_val_key(m[-1]["val"]["pos_rmse_m"])) if m and m[-1].get("val") else None,
                   final_train_loss=m[-1]["train"]["total"] if m else None,
                   train_s=sum(e["epoch_s"] for e in m), total_wall_s=wall,
                   mean_windows_per_s=float(np.mean([e["windows_per_s"] for e in m])) if m else None,
                   median_windows_per_s=float(np.median([e["windows_per_s"] for e in m])) if m else None,
                   peak_gpu_mem_mb=max([*gmem, *(x["gpu_peak_reserved_mb"] or 0 for x in ep_res)], default=None) or None,
                   peak_rss_mb=max([*rss_pk, *(x["rss_peak_mb"] or 0 for x in ep_res)], default=None) or None,
                   mean_cpu_pct=float(np.mean(cpu)) if cpu else (float(np.mean([x["cpu_pct"] for x in ep_res])) if ep_res else None),
                   sampled_s=(max(t_res) - min(t_res)) if t_res else None, started=pv.get("started", ""), finished=(sm or {}).get("finished", ""),
                   device=(sm or {}).get("device") or (res[0]["gpu"] and f"cuda:{res[0]['gpu']}" if res else ""))
        rows.append(row)
    return rows


def eval_tables(eval_dir: Path | None, tags: Sequence[str]) -> dict[str, dict]:
    """{"v05": results, "v06": results, "v05_on_v06": results, ...} from evaluation.run JSON files."""
    out = {}
    if not eval_dir or not eval_dir.exists():
        return out
    for p in sorted(eval_dir.glob("*.json")):
        stem = p.stem
        if stem in tags or any(stem.startswith(f"{t}_on_") for t in tags):
            d = _read_json(p)
            if isinstance(d, dict):
                out[stem] = d
    return out


def load_analyses(eval_dir: Path | None, tags: Sequence[str]) -> dict[str, dict[str, dict]]:
    """{"regime": {"v05": json, ...}, "anchor": ..., "residual": ..., "shuffle": ...} from ``analysis_<kind>_<tag>.json``."""
    out: dict[str, dict[str, dict]] = {}
    for kind in ("regime", "anchor", "residual", "shuffle"):
        for t in tags:
            d = _read_json(eval_dir / f"analysis_{kind}_{t}.json") if eval_dir else None
            if d:
                out.setdefault(kind, {})[t] = d
    return out


def analysis_rows(an: Mapping[str, Mapping[str, dict]]) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {"regime": [], "gap": [], "anchor": [], "residual": [], "shuffle": []}
    for t, d in an.get("regime", {}).items():
        for reg, b in d.get("regimes", {}).items():
            for m, c in b["pos_rmse_m"].items():
                rows["regime"].append(dict(tag=t, regime=reg, episodes=b["n_episodes"], windows=b["n_windows"], model=m, **{f"rmse_{k}": c.get(str(k)) for k in (1, 5, 20)}))
            for pair, g in b.get("gaps", {}).items():
                ks = [int(x) for x in g["ks"]]; i = ks.index(max(ks)); ci = g.get("gap_ci95")
                rows["gap"].append(dict(tag=t, regime=reg, pair=pair, k=ks[i], rmse_a=g["rmse_a"][i], rmse_b=g["rmse_b"][i], gap=g["gap"][i],
                                        ci_low=ci[0][i] if ci else None, ci_high=ci[1][i] if ci else None, episodes=g["n_episodes"]))
    for t, d in an.get("anchor", {}).items():
        for m, r in d.get("models", {}).items():
            ks = sorted(int(k) for k in r["open_loop"]); k5, kl = 5 if 5 in ks else ks[0], ks[-1]
            row = dict(tag=t, model=m, windows=r.get("n_windows"), open_k5=r["open_loop"].get(str(k5)), open_last=r["open_loop"].get(str(kl)), k_last=kl)
            for K, a in r.get("anchored", {}).items():
                row[f"every{K}_last"] = a["rmse"].get(str(kl))
            b = r.get("phi_vs_real_imu_boundary") or {}
            row["phi_vs_real_imu_k5"] = b.get(str(k5)); row["phi_vs_real_imu_last"] = b.get(str(kl))
            rows["anchor"].append(row)
    for t, d in an.get("residual", {}).items():
        for m, regs in d.get("models", {}).items():
            for reg, v in regs.items():
                dg = v.get("degradation_ratio", {})
                rows["residual"].append(dict(tag=t, model=m, regime=reg, windows=v.get("n_windows"), r_acc_over_specific_force_k1=(v.get("ratio_r_acc_to_f_spec") or [None])[0],
                                             r_ang_over_phys_k1=(v.get("ratio_r_ang_to_f_ang") or [None])[0], rmse_ratio_residual_zeroed_last=dg.get(str(max((int(k) for k in dg), default=0))) if dg else None))
    for t, d in an.get("shuffle", {}).items():
        for m, v in d.get("models", {}).items():
            if v.get("n_windows"):
                jr = v["jump_ratio"]; ks = sorted(int(k) for k in jr)
                rows["shuffle"].append(dict(tag=t, model=m, windows=v["n_windows"], jump_k1=jr.get(str(ks[0])), jump_k5=jr.get("5"), jump_last=jr.get(str(ks[-1])), k_last=ks[-1]))
    return rows


def fig_regime(an, tags, out: Path, k: int = 20) -> str | None:
    d = an.get("regime", {})
    if not all(t in d for t in tags):
        return None
    plt = _plt()
    regs = [r for r in ("wind", "no_wind") if all(r in d[t].get("regimes", {}) for t in tags)]
    if not regs:
        return None
    models = list(d[tags[0]]["regimes"][regs[0]]["pos_rmse_m"])
    fig, axes = plt.subplots(1, len(regs), figsize=(5.4 * len(regs), 3.8), squeeze=False, constrained_layout=True, sharey=True)
    for ax, reg in zip(axes[0], regs):
        w = 0.4
        for j, t in enumerate(tags):
            b = d[t]["regimes"][reg]
            ys = [b["pos_rmse_m"].get(m, {}).get(str(k), np.nan) for m in models]
            ax.bar(np.arange(len(models)) + (j - 0.5) * w, ys, w * 0.92, color=_color(t, tags))
        n = d[tags[0]]["regimes"][reg]["n_episodes"]
        ax.set_xticks(np.arange(len(models))); ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=7); ax.set_yscale("log")
        ax.set_title(f"{reg.replace('_', ' ')} episodes ({n} episodes)", fontsize=9); ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel(f"position RMSE at {k} steps [m] (log)")
    _legend(fig, tags)
    p = out / "figures" / "analysis_wind_vs_no_wind.png"; p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=130); plt.close(fig)
    return "analysis_wind_vs_no_wind.png"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def fmt(v: Any, nd: int = 4) -> str:
    if v is None or v == "":
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1e5 else f"{v:.3e}"
    return str(v)


def md_table(rows: list[dict], cols: Sequence[tuple[str, str]], nd: int = 4) -> str:
    head = "| " + " | ".join(h for _, h in cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(r.get(k), nd) for k, _ in cols) + " |" for r in rows) + "\n"


def hms(s: float | None) -> str:
    if s is None:
        return "–"
    s = int(round(s)); return f"{s // 3600}h {s % 3600 // 60:02d}m {s % 60:02d}s"


# ------------------------------------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.8})
    return plt


def _color(tag: str, tags: Sequence[str]) -> str:
    return COLORS[list(tags).index(tag) % 2]


def _legend(fig, tags, y=0.995):
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=_color(t, tags), lw=3, label=DATASET_NOTE.get(t, t)) for t in tags]
    try:
        fig.legend(handles=h, loc="outside upper right", ncol=len(tags), fontsize=9)       # reserves its own row (constrained layout)
    except ValueError:
        fig.legend(handles=h, loc="upper right", ncol=len(tags), fontsize=9)


def _plain_log_ticks(ax):
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g")); ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3g"))
    ax.tick_params(axis="y", which="minor", labelsize=6.5)


def fig_curves(runs, tags, out: Path, what: str) -> str | None:
    plt = _plt()
    kinds = [k for k in KINDS if any((t, k) in runs and runs[(t, k)]["metrics"] for t in tags)]
    if not kinds:
        return None
    ncol = 4; nrow = int(np.ceil(len(kinds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow + 0.5), squeeze=False, constrained_layout=True)
    for ax in axes.flat[len(kinds):]:
        ax.axis("off")
    for ax, k in zip(axes.flat, kinds):
        for t in tags:
            m = runs.get((t, k), {}).get("metrics") or []
            if not m:
                continue
            if what == "loss":
                x = [e["epoch"] for e in m]; y = [e["train"]["total"] for e in m]
            elif what == "val":
                pts = [(e["epoch"], e["val"]["pos_rmse_m"][_val_key(e["val"]["pos_rmse_m"])]) for e in m if e.get("val")]
                x, y = zip(*pts) if pts else ([], [])
            else:
                x = [e["epoch"] for e in m]; y = [e["windows_per_s"] for e in m]
            # the first tag is drawn thick underneath, the second thin on top: identical curves show as an orange line inside a blue one
            ax.plot(x, y, color=_color(t, tags), lw=3.0 if t == tags[0] else 1.4, zorder=2 if t == tags[0] else 3)
        m0 = next((runs[(t, k)]["metrics"] for t in tags if (t, k) in runs and runs[(t, k)]["metrics"]), [])
        prev = None
        for e in m0:                                                           # curriculum stage boundaries (horizon changes)
            if prev is not None and e["horizon"] != prev:
                ax.axvline(e["epoch"] - 0.5, color=INK2, lw=0.7, ls=":")
                ax.text(e["epoch"] - 0.4, ax.get_ylim()[1], f"H={e['horizon']}", fontsize=7, color=INK2, va="top")
            prev = e["horizon"]
        if what != "tp":
            ax.set_yscale("log"); _plain_log_ticks(ax)
        ax.set_title(LABEL[k], fontsize=9); ax.set_xlabel("epoch")
    ylab = {"loss": "training loss (log)", "val": "validation position RMSE at the last validated step [m] (log)", "tp": "throughput [windows/s]"}[what]
    fig.supylabel(ylab, color=INK2, fontsize=9)
    _legend(fig, tags)
    name = {"loss": "training_loss.png", "val": "validation_error.png", "tp": "throughput_by_epoch.png"}[what]
    p = out / "figures" / name; p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=130); plt.close(fig)
    return name


def _grouped_bars(rows, tags, value, ylabel, title, name, out: Path, fmt_fn=None, log=False, note=None) -> str | None:
    plt = _plt()
    kinds = [k for k in KINDS if any(r["kind"] == k and r.get(value) is not None for r in rows)]
    if not kinds:
        return None
    fig, ax = plt.subplots(figsize=(max(6.5, 1.15 * len(kinds) + 2), 3.9), constrained_layout=True)
    w = 0.38
    for j, t in enumerate(tags):
        ys = []
        for k in kinds:
            r = next((r for r in rows if r["tag"] == t and r["kind"] == k), None)
            ys.append(r.get(value) if r and r.get(value) is not None else np.nan)
        x = np.arange(len(kinds)) + (j - (len(tags) - 1) / 2) * w
        ax.bar(x, ys, w * 0.92, color=_color(t, tags), label=DATASET_NOTE.get(t, t))   # surface gap between neighbouring bars via the 0.92 width
        for xi, yi in zip(x, ys):
            if not np.isnan(yi):
                ax.text(xi, yi, (fmt_fn or (lambda v: f"{v:.3g}"))(yi), ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(np.arange(len(kinds))); ax.set_xticklabels([LABEL[k].replace(" ", "\n", 1) for k in kinds], fontsize=7.5)
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left")
    if log:
        ax.set_yscale("log")
    ax.margins(y=0.18); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=len(tags), fontsize=8)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color=INK2)
    p = out / "figures" / name; p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=130); plt.close(fig)
    return name


def _parse_utc(txt: str, offset_h: float = 0.0) -> float | None:
    import calendar, time as _t
    try:
        return calendar.timegm(_t.strptime(txt, "%Y-%m-%d %H:%M:%S")) - offset_h * 3600
    except (ValueError, TypeError):
        return None


def fig_timeline(runs_rows, samples, gpu_samples, tags, out: Path, tz_offset_h: float = 0.0) -> str | None:
    """Scheduling lanes (one row per run, grouped by GPU) and the CPU use of our processes over time."""
    if not samples:
        return None
    plt = _plt()
    by = defaultdict(list)
    for s in samples:
        by[s["run"]].append(s)
    spans = {}
    for r in by:
        ts = [float(x["t_unix"]) for x in by[r]]
        row = next((x for x in runs_rows if f"{x['tag']}_{x['kind']}" == r), {})
        a0, a1 = _parse_utc(row.get("started", ""), tz_offset_h), _parse_utc(row.get("finished", ""), tz_offset_h)
        start = a0 if a0 and min(ts) - 3600 < a0 <= min(ts) + 60 else min(ts)            # real start when its clock agrees with the sampler's
        end = a1 if a1 and a1 >= max(ts) - 60 and a1 < max(ts) + 3600 else max(ts)
        spans[r] = (start, end)
    runs_sorted = sorted(by, key=lambda r: (int(by[r][0]["gpu"]), spans[r][0]))
    t0 = min(v[0] for v in spans.values())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 0.3 * len(runs_sorted) + 5.0), gridspec_kw={"height_ratios": [max(1.0, len(runs_sorted) * 0.3), 2.2]},
                                   constrained_layout=True, sharex=True)
    for i, r in enumerate(runs_sorted):
        a, b = spans[r]
        ax1.barh(i, (b - a) / 60, left=(a - t0) / 60, height=0.62, color=_color(r.split("_")[0], tags))
    ax1.set_yticks(range(len(runs_sorted))); ax1.set_yticklabels([f"{r} (GPU {by[r][0]['gpu']})" for r in runs_sorted], fontsize=7)
    ax1.set_title("Runs over time (bar = start to end of the run, colour = dataset, row = run and GPU)", loc="left"); ax1.invert_yaxis(); ax1.grid(axis="y", visible=False)
    per = defaultdict(list)                                                           # mean per (time bucket, process), then sum over processes
    for s in samples:
        if s["cpu_pct"] not in ("", None):
            per[(int((float(s["t_unix"]) - t0) // 10), s["run"])].append(float(s["cpu_pct"]))
    tot = defaultdict(float)
    for (bk, _), v in per.items():
        tot[bk] += float(np.mean(v)) / 100.0
    xs = sorted(tot)
    ax2.fill_between([(x * 10 + 5) / 60 for x in xs], [tot[x] for x in xs], color=COLORS[0], alpha=0.35, lw=0)
    ax2.plot([(x * 10 + 5) / 60 for x in xs], [tot[x] for x in xs], color=COLORS[0])
    ax2.set_ylabel("CPU cores used"); ax2.set_xlabel("minutes since the first run started"); ax2.set_title("CPU use of our training processes (sum over processes, 10 s buckets)", loc="left")
    p = out / "figures" / "run_timeline_cpu.png"; p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=130); plt.close(fig)
    return "run_timeline_cpu.png"


def fig_gpu(gpu_samples, out: Path, used: Sequence[int] | None = None) -> str | None:
    if not gpu_samples:
        return None
    if used:
        gpu_samples = [s for s in gpu_samples if int(s["gpu"]) in set(used)]
    plt = _plt()
    t0 = min(float(s["t_unix"]) for s in gpu_samples)
    gpus = sorted({int(s["gpu"]) for s in gpu_samples})
    fig, axes = plt.subplots(2, len(gpus), figsize=(3.6 * len(gpus), 4.6), squeeze=False, constrained_layout=True, sharex="col")
    for j, g in enumerate(gpus):
        rows = [s for s in gpu_samples if int(s["gpu"]) == g]
        x = [(float(s["t_unix"]) - t0) / 60 for s in rows]
        axes[0][j].plot(x, [float(s["util_pct"]) for s in rows], color=COLORS[0]); axes[0][j].set_ylim(0, 100)
        axes[0][j].set_title(f"GPU {g}: utilisation [%]", loc="left"); axes[1][j].plot(x, [float(s["mem_used_mb"]) / 1024 for s in rows], color=COLORS[0])
        axes[1][j].set_title(f"GPU {g}: memory used [GiB]", loc="left"); axes[1][j].set_xlabel("minutes")
    fig.suptitle("GPU load while training (whole GPU, including any other user's processes)", x=0.01, ha="left", fontsize=9, color=INK2)
    p = out / "figures" / "gpu_utilisation_memory.png"; p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=130); plt.close(fig)
    return "gpu_utilisation_memory.png"


def _models_in(results: Mapping[str, Mapping]) -> list[str]:
    ms = list(results)
    return [m for m in FIXED + KINDS if m in ms] + [m for m in ms if m not in FIXED + KINDS]


def fig_eval_roles(eval_res, tags, out: Path, k: int = 20, roles: Sequence[str] | None = None) -> str | None:
    if not all(t in eval_res for t in tags):
        return None
    plt = _plt()
    r0 = eval_res[tags[0]]; models = _models_in(r0)
    roles = list(roles or [r for r in next(iter(r0.values())).keys() if next(iter(r0.values()))[r].get("n_windows")])
    ncol = 4; nrow = int(np.ceil(len(roles) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.6 * nrow + 0.4), squeeze=False, constrained_layout=True)
    for ax in axes.flat[len(roles):]:
        ax.axis("off")
    for ax, role in zip(axes.flat, roles):
        w = 0.4
        for j, t in enumerate(tags):
            ys = [(eval_res[t].get(m, {}).get(role) or {}).get("pos_rmse_m", {}).get(str(k), np.nan) for m in models]
            ax.bar(np.arange(len(models)) + (j - 0.5) * w, ys, w * 0.92, color=_color(t, tags))
        ax.set_xticks(np.arange(len(models))); ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=6)
        ax.set_yscale("log"); ax.set_title(role, fontsize=9); ax.grid(axis="x", visible=False)
    fig.supylabel(f"position RMSE at {k} steps [m] (log)", color=INK2, fontsize=9)
    _legend(fig, tags)
    p = out / "figures" / "eval_pos_rmse_by_role.png"; p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=130); plt.close(fig)
    return "eval_pos_rmse_by_role.png"


def fig_error_vs_step(eval_res, tags, out: Path, role: str = "test_id") -> str | None:
    if not all(t in eval_res for t in tags):
        return None
    plt = _plt()
    models = _models_in(eval_res[tags[0]])
    ncol = 4; nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.7 * nrow + 0.4), squeeze=False, constrained_layout=True, sharey=True)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    for ax, m in zip(axes.flat, models):
        for t in tags:
            v = (eval_res[t].get(m, {}).get(role) or {}).get("pos_rmse_m", {})
            ks = sorted(int(x) for x in v)
            ax.plot(ks, [v[str(x)] for x in ks], marker="o", ms=4, color=_color(t, tags))
        ax.set_yscale("log"); ax.set_xscale("log"); ax.set_title(LABEL.get(m, m), fontsize=8.5); ax.set_xlabel("rollout step")
    fig.supylabel(f"position RMSE on {role} [m] (log)", color=INK2, fontsize=9)
    _legend(fig, tags)
    p = out / "figures" / "eval_error_vs_step.png"; p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=130); plt.close(fig)
    return "eval_error_vs_step.png"


def fig_cross(eval_res, tags, out: Path, role: str = "test_id", k: int = 20) -> str | None:
    """Heatmap: rows = models, columns = train dataset -> evaluation dataset (sequential blue, numbers in the cells)."""
    if len(tags) != 2:
        return None
    a, b = tags
    cols = [(a, a), (a, b), (b, b), (b, a)]
    keys = {(a, a): a, (b, b): b, (a, b): f"{a}_on_{b}", (b, a): f"{b}_on_{a}"}
    if not all(keys[c] in eval_res for c in cols):
        return None
    plt = _plt()
    models = [m for m in _models_in(eval_res[a]) if m in KINDS]
    M = np.array([[(eval_res[keys[c]].get(m, {}).get(role) or {}).get("pos_rmse_m", {}).get(str(k), np.nan) for c in cols] for m in models])
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(models) + 1.6), constrained_layout=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#3987e5", "#0d366b"])
    im = ax.imshow(np.log10(M), cmap=cmap, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"trained {x}\nevaluated {y}" for x, y in cols], fontsize=8)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([LABEL[m] for m in models], fontsize=8); ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white" if np.log10(M[i, j]) > np.nanmedian(np.log10(M)) else INK)
    ax.set_title(f"Cross-evaluation: position RMSE at {k} steps on {role} [m] (colour: log scale)", loc="left", fontsize=9)
    p = out / "figures" / "cross_evaluation.png"; p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=130); plt.close(fig)
    return "cross_evaluation.png"


# ------------------------------------------------------------------------------------------------ evaluation tables
def eval_rows(eval_res: Mapping[str, Mapping], tag: str, ks=(1, 5, 10, 20)) -> list[dict]:
    rows = []
    for m in _models_in(eval_res[tag]):
        for role, r in eval_res[tag][m].items():
            if not r.get("n_windows"):
                continue
            row = dict(tag=tag, model=m, role=role, windows=r["n_windows"], episodes=r["n_episodes"])
            for k in ks:
                row[f"pos_rmse_{k}"] = r["pos_rmse_m"].get(str(k))
            row["att_rmse_deg_last"] = r["att_rmse_deg"].get(str(max(int(x) for x in r["att_rmse_deg"])))
            row["reliable_horizon"] = r.get("reliable_horizon_mean")
            rows.append(row)
    return rows


def delta_rows(eval_res, tags, k=20) -> list[dict]:
    if len(tags) != 2 or not all(t in eval_res for t in tags):
        return []
    a, b = tags
    rows = []
    for m in _models_in(eval_res[a]):
        for role, ra in eval_res[a][m].items():
            rb = (eval_res[b].get(m) or {}).get(role)
            if not ra.get("n_windows") or not rb or not rb.get("n_windows"):
                continue
            va, vb = ra["pos_rmse_m"].get(str(k)), rb["pos_rmse_m"].get(str(k))
            if va and vb:
                rows.append(dict(model=m, role=role, uses_imu="yes" if m in USES_IMU else "no", **{f"rmse_{k}_{a}": va, f"rmse_{k}_{b}": vb, "change_pct": 100.0 * (vb - va) / va}))
    return rows


# ------------------------------------------------------------------------------------------------ report
CSS = """body{font-family:system-ui,Segoe UI,Roboto,sans-serif;max-width:1180px;margin:0 auto;padding:24px 16px;background:#fcfcfb;color:#0b0b0b;line-height:1.5}
h1{font-size:1.6rem}h2{margin-top:2.2rem;border-bottom:1px solid #e4e3df;padding-bottom:.25rem}h3{margin-top:1.4rem}table{border-collapse:collapse;font-size:.82rem;margin:.6rem 0;display:block;overflow-x:auto}
th,td{border:1px solid #e4e3df;padding:3px 7px;text-align:right;white-space:nowrap}th{background:#f0efec}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}
img{max-width:100%;height:auto;border:1px solid #e4e3df;margin:.4rem 0}code{background:#f0efec;padding:1px 4px;border-radius:3px}.note{color:#52514e;font-size:.88rem}"""


def md_to_html(md: str, out: Path) -> str:
    """Minimal Markdown -> HTML for this report (headings, tables, images, paragraphs, bullets, code spans); images are embedded."""
    lines, out_html, i = md.splitlines(), [], 0
    def inline(t: str) -> str:
        t = html.escape(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t); t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
        return t
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", ln)
            p = out / m.group(2)
            if p.exists():
                out_html.append(f'<img alt="{html.escape(m.group(1))}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">')
        elif ln.startswith("|"):
            blk = []
            while i < len(lines) and lines[i].startswith("|"):
                blk.append(lines[i]); i += 1
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in blk if not re.match(r"^\|[-| ]+\|$", r)]
            t = "<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</tbody></table>"
            out_html.append(t); continue
        elif ln.startswith("# "):
            out_html.append(f"<h1>{inline(ln[2:])}</h1>")
        elif ln.startswith("## "):
            out_html.append(f"<h2>{inline(ln[3:])}</h2>")
        elif ln.startswith("### "):
            out_html.append(f"<h3>{inline(ln[4:])}</h3>")
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>"); i += 1
            out_html.append("<ul>" + "".join(items) + "</ul>"); continue
        elif ln.strip():
            out_html.append(f'<p class="{"note" if ln.startswith("_") else ""}">{inline(ln.strip("_"))}</p>')
        i += 1
    return f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Baseline comparison report</title><style>{CSS}</style></head><body>{''.join(out_html)}</body></html>"


def build(exp_root: Path, tags: Sequence[str], out: Path, eval_dir: Path | None, res_dir: Path | None, tz_offset_h: float = 0.0) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(exp_root, tags)
    samples, gpu_samples = load_samples(res_dir)
    rows = runs_table(runs, samples)
    write_csv(out / "tables" / "runs.csv", rows)
    ev = eval_tables(eval_dir, tags)
    figs: dict[str, str | None] = {}
    figs["loss"] = fig_curves(runs, tags, out, "loss"); figs["val"] = fig_curves(runs, tags, out, "val"); figs["tp"] = fig_curves(runs, tags, out, "tp")
    figs["best"] = _grouped_bars(rows, tags, "best_val_pos_rmse_m", "best validation position RMSE [m]", "Best validation error (validated horizon per run)", "best_validation_error.png", out, lambda v: f"{v:.3f}")
    figs["time"] = _grouped_bars(rows, tags, "total_wall_s", "total wall time [min]", "Total training wall time", "training_time.png", out, lambda v: f"{v/60:.0f} min") if any(r.get("total_wall_s") for r in rows) else None
    if figs["time"] is None:                                                        # runs without a summary yet: sum of epoch times
        for r in rows:
            r["_train_min"] = r["train_s"] / 60 if r.get("train_s") else None
        figs["time"] = _grouped_bars(rows, tags, "_train_min", "training time (sum of epoch times) [min]", "Training time", "training_time.png", out, lambda v: f"{v:.0f} min")
    figs["tpbar"] = _grouped_bars(rows, tags, "median_windows_per_s", "median throughput [windows/s]", "Training throughput (median over epochs)", "throughput.png", out, lambda v: f"{v:.0f}",
                                  note="Throughput depends on the horizon stage and on the other jobs sharing the GPU and CPU; compare across kinds with care.")
    figs["gmem"] = _grouped_bars(rows, tags, "peak_gpu_mem_mb", "peak GPU memory of the process [MiB]", "Peak GPU memory per run (nvidia-smi, includes the CUDA context)", "peak_gpu_memory.png", out, lambda v: f"{v:.0f}")
    figs["rss"] = _grouped_bars(rows, tags, "peak_rss_mb", "peak resident memory [MiB]", "Peak CPU (resident) memory per run", "peak_cpu_memory.png", out, lambda v: f"{v:.0f}")
    figs["cpu"] = _grouped_bars(rows, tags, "mean_cpu_pct", "mean CPU use [% of one core]", "Mean CPU use per run", "mean_cpu.png", out, lambda v: f"{v:.0f}")
    figs["timeline"] = fig_timeline(rows, samples, gpu_samples, tags, out, tz_offset_h)
    figs["gpu"] = fig_gpu(gpu_samples, out, sorted({int(s["gpu"]) for s in samples if str(s["gpu"]).isdigit()}))
    figs["roles"] = fig_eval_roles(ev, tags, out); figs["steps"] = fig_error_vs_step(ev, tags, out); figs["cross"] = fig_cross(ev, tags, out)
    # evaluation tables
    for t in tags:
        if t in ev:
            write_csv(out / "tables" / f"evaluation_{t}.csv", eval_rows(ev, t))
    deltas = delta_rows(ev, tags)
    write_csv(out / "tables" / "v06_vs_v05_change.csv", deltas) if deltas else None
    an = load_analyses(eval_dir, tags)
    arows = analysis_rows(an)
    for name, rr in arows.items():
        if rr:
            write_csv(out / "tables" / f"analysis_{name}.csv", rr)
    figs["regime"] = fig_regime(an, tags, out)
    prov = next((r["prov"] for r in runs.values() if r["prov"]), {})
    cfg = next((r["config"].get("train", {}) for r in runs.values() if r["config"]), {})

    md = ["# Baseline comparison: pilot v0.5.0 (raw Isaac IMU) vs pilot v0.6.0 (corrected IMU)", ""]
    n_done = sum(r["status"] == "finished" for r in rows)
    md += [f"_Generated from {len(rows)} runs ({n_done} finished). All numbers are from one seed on a 147-episode pilot; see Limitations._", ""]
    md += ["## 1. Setup and metadata", "",
           f"- Datasets: {', '.join(f'`{t}` = {DATASET_NOTE.get(t, t)}' for t in tags)}. Same plan (episodes, seeds, wind, vehicles); v0.6.0 differs by the IMU transport correction and the determinism fixes.",
           f"- Training configuration: curriculum horizons {cfg.get('horizon_stages')}, at most {cfg.get('stage_max_epochs')} epochs per stage, {cfg.get('epochs')} epochs in total, early stopping patience {cfg.get('patience')}, validated at {cfg.get('val_horizon')} steps, "
           f"IMU noise `{cfg.get('noise')}` (training) / `{cfg.get('val_noise')}` (validation), seed {cfg.get('seed')}, batch {cfg.get('batch_size')}, lr {cfg.get('lr')}, history {cfg.get('history')} steps.",
           f"- Software: torch {prov.get('torch', '?')}, git `{str(prov.get('git', '?'))[:10]}`. Hardware: NVIDIA RTX A6000 GPUs (shared host, 96 CPU cores, other users' jobs running).",
           "- Metrics: position RMSE = sqrt(mean over windows of the squared Euclidean error) at rollout step k (primary); mean Euclidean error is in the CSV files of the evaluation.", ""]
    md += ["## 2. Training runs", "", md_table(rows, [("tag", "run set"), ("model", "model"), ("params", "parameters"), ("status", "status"), ("epochs", "epochs"), ("final_horizon", "last horizon"),
                                                      ("stop_reason", "stop"), ("best_epoch", "best epoch"), ("best_val_pos_rmse_m", "best val RMSE [m]"), ("final_train_loss", "final train loss")])]
    md += ["Parameter counts of B2 and B4 are matched to B3 within ±10 %; `B4 (B3 width)` has B3's residual width and is not matched.", ""]
    for key, title, alt in (("loss", "Training loss", "training loss by epoch"), ("val", "Validation error", "validation error by epoch")):
        if figs[key]:
            md += [f"### {title}", "", f"![{alt}](figures/{figs[key]})", ""]
    if figs["best"]:
        md += ["![best validation error](figures/best_validation_error.png)", ""]
    md += ["_Dotted vertical lines mark curriculum stage changes (horizon H); the loss is not comparable across horizons._", ""]
    md += ["## 3. Time, throughput and resources", "",
           md_table([dict(r, wall=hms(r["total_wall_s"] if r.get("total_wall_s") else r.get("train_s")), mean_tp=r["mean_windows_per_s"]) for r in rows],
                    [("tag", "run set"), ("model", "model"), ("wall", "training time (h m s)"), ("mean_windows_per_s", "mean windows/s"), ("median_windows_per_s", "median windows/s"),
                     ("peak_gpu_mem_mb", "peak GPU mem [MiB]"), ("peak_rss_mb", "peak CPU mem [MiB]"), ("mean_cpu_pct", "mean CPU [% core]"), ("device", "device")], nd=4),
           "_Training time is the wall time of the run when `summary.json` has it, otherwise the sum of the epoch times (training only, validation excluded). GPU memory is the process memory seen by nvidia-smi "
           "(PyTorch cache plus the CUDA context); CPU memory is the kernel-tracked peak resident set (VmHWM). Resource values for these runs come from an external sampler (2 s) that started a few minutes after "
           "the runs did, so the first curriculum stages are not sampled; the peak resident memory is exact (kernel peak), the GPU peak is the maximum of the samples. Runs started after the trainer gained "
           "per-epoch resource logging also record them in `metrics.jsonl`._", ""]
    for key, alt in (("time", "training time per run"), ("tpbar", "throughput per run"), ("gmem", "peak GPU memory per run"), ("rss", "peak CPU memory per run"), ("cpu", "mean CPU per run"),
                     ("tp", "throughput by epoch"), ("timeline", "run timeline and CPU use"), ("gpu", "GPU utilisation and memory")):
        if figs[key]:
            md += [f"![{alt}](figures/{figs[key] if key not in ('tp',) else 'throughput_by_epoch.png'})", ""]
    if any(t in ev for t in tags):
        md += ["## 4. Evaluation", "", "Rollouts from the true state at the end of the 10-step context window; nominal IMU noise; same windows for every model. B0, B1 and B6a are not trained.", ""]
        for t in tags:
            if t in ev:
                r = [x for x in eval_rows(ev, t) if x["role"] in ("val_id", "test_id", "test_ood_magnitude", "test_ood_direction", "test_ood_temporal", "test_ood_gust", "test_ood_physics", "test_ood_trajectory")]
                md += [f"### Position RMSE [m], {DATASET_NOTE.get(t, t)}", "", md_table(r, [("model", "model"), ("role", "role"), ("episodes", "episodes"), ("pos_rmse_1", "k=1"), ("pos_rmse_5", "k=5"), ("pos_rmse_10", "k=10"),
                                                                                              ("pos_rmse_20", "k=20"), ("att_rmse_deg_last", "attitude RMSE at last k [deg]"), ("reliable_horizon", "reliable horizon [steps]")]), ""]
        for key, alt in (("roles", "position RMSE at 20 steps by role"), ("steps", "error versus rollout step")):
            if figs[key]:
                md += [f"![{alt}](figures/{figs[key]})", ""]
        if deltas:
            md += [f"### Change from {tags[0]} to {tags[1]} (position RMSE at 20 steps, same model, same windows)", "",
                   md_table(sorted(deltas, key=lambda d: (d["uses_imu"] != "yes", d["model"], d["role"])), [("model", "model"), ("role", "role"), ("uses_imu", "reads the IMU"),
                                                                                                          (f"rmse_20_{tags[0]}", f"{tags[0]} [m]"), (f"rmse_20_{tags[1]}", f"{tags[1]} [m]"), ("change_pct", "change [%]")]),
                   "_Models that never read the IMU (B1, B4, B4 at B3 width, B6a) are the control: their change should be near zero._", ""]
        if figs["cross"]:
            md += ["### Cross-evaluation", "", "![cross-evaluation](figures/cross_evaluation.png)", "",
                   "_A model trained on one dataset and evaluated on the other: shows how much a model that reads the IMU depends on the sensor convention._", ""]
    if any(arows.values()):
        md += ["## 5. Analyses", "", "Wind vs no-wind, paired gaps, re-observation, residual diagnostics and shuffled-u (see `.idea/specs/model_specification.md` section 11.4). Each tag is analysed on its own dataset.", ""]
        if arows["regime"]:
            if figs["regime"]:
                md += ["![wind vs no wind](figures/analysis_wind_vs_no_wind.png)", ""]
            md += ["### Position RMSE by wind regime [m]", "", md_table(arows["regime"], [("tag", "run set"), ("regime", "regime"), ("episodes", "episodes"), ("model", "model"), ("rmse_1", "k=1"), ("rmse_5", "k=5"), ("rmse_20", "k=20")]), ""]
        if arows["gap"]:
            md += ["### Paired gaps (RMSE of the first model minus the second, same windows; 95 % interval by episode bootstrap)", "",
                   md_table(arows["gap"], [("tag", "run set"), ("regime", "regime"), ("pair", "a:b"), ("k", "k"), ("rmse_a", "RMSE a [m]"), ("rmse_b", "RMSE b [m]"), ("gap", "gap [m]"), ("ci_low", "ci low"), ("ci_high", "ci high"), ("episodes", "episodes")]),
                   "_Positive: the first model is worse. B6a/B6b use the total recorded wrench, so even without wind their gap includes vehicle-motion drag. The no-wind sample is small._", ""]
        if arows["anchor"]:
            md += ["### Re-observation with the real IMU (latent models)", "", md_table(arows["anchor"], [("tag", "run set"), ("model", "model"), ("open_k5", "open loop k=5 [m]"), ("open_last", "open loop last k [m]"), ("k_last", "last k"),
                                                                                                          ("every1_last", "re-observe every 1"), ("every5_last", "every 5"), ("every10_last", "every 10"),
                                                                                                          ("phi_vs_real_imu_k5", "cost of φ vs real IMU, k=5"), ("phi_vs_real_imu_last", "cost, last k")]), ""]
        if arows["residual"]:
            md += ["### Residual diagnostics (B3)", "", md_table(arows["residual"], [("tag", "run set"), ("model", "model"), ("regime", "regime"), ("windows", "windows"), ("r_acc_over_specific_force_k1", "|d_θ,v̇| / specific force, k=1"),
                                                                                         ("r_ang_over_phys_k1", "|d_θ,ω̇| / |ω̇_phys|, k=1"), ("rmse_ratio_residual_zeroed_last", "RMSE × when d_θ is zeroed")]), ""]
        if arows["shuffle"]:
            md += ["### Shuffled-u test (jump = RMSE with the sibling's actions / RMSE with the original actions)", "", md_table(arows["shuffle"], [("tag", "run set"), ("model", "model"), ("windows", "windows"), ("jump_k1", "k=1"), ("jump_k5", "k=5"), ("jump_last", "last k")]),
                   "_Read each model relative to B1 (physics only): u legitimately drives the dynamics, and sibling controllers' commands differ little in this data, so the test has low power._", ""]
    md += ["## 6. Limitations", "",
           "- One seed per model, 63 training episodes (pilot): differences of a few percent are within what a second seed could change. A statement about which model is better needs more seeds and the full dataset.",
           "- Short curriculum (horizons up to 20 steps), early stopping on one validation split; no hyper-parameter tuning; λ_d and the loss floors are proposed values.",
           "- Throughput and wall time were measured with up to 9 jobs sharing 3 GPUs and a loaded shared host; they are not stand-alone speeds. GPU utilisation is for the whole GPU, including other users.",
           "- Roles with few independent groups (several OOD roles have 2–4) give wide uncertainty; episode-level bootstrap intervals are in the evaluation JSON.", "",
           "## 7. Files", "", "- `tables/*.csv`: run table, evaluation tables, v0.5→v0.6 change table. `figures/*.png`: all figures. `report.html`: this report with the images embedded.", ""]
    (out / "REPORT.md").write_text("\n".join(md))
    (out / "report.html").write_text(md_to_html("\n".join(md), out))
    return out / "report.html"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--experiments", type=Path, default=Path("outputs/experiments"))
    p.add_argument("--tags", nargs="+", default=["v05", "v06"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval", type=Path, default=Path("outputs/eval"))
    p.add_argument("--resources", type=Path, default=None)
    p.add_argument("--tz-offset-hours", type=float, default=0.0, help="UTC offset of the clock that wrote provenance.json / summary.json (the container runs in UTC)")
    a = p.parse_args(argv)
    print(build(a.experiments, a.tags, a.out, a.eval, a.resources or a.out, a.tz_offset_hours))


if __name__ == "__main__":
    main()
