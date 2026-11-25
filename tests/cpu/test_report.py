"""Report generator: tables, figures, HTML/Markdown from synthetic run directories."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from physiworld_uav.evaluation import report as R

pytest.importorskip("matplotlib")


def _run(exp: Path, tag: str, kind: str, offset: float, finished: bool = True, resources: bool = False):
    d = exp / f"{tag}_{kind}"
    d.mkdir(parents=True)
    rec = []
    for i, h in enumerate([1, 1, 5, 5, 10, 10, 20, 20]):
        rec.append({"epoch": i + 1, "step": 10 * (i + 1), "horizon": h, "stage": 0, "lr": 1e-3, "epoch_s": 10.0 + i, "windows_per_s": 1000.0 - 20 * i,
                    "train": {"total": 100.0 / (i + 1), "state": 1.0, "roll": 2.0},
                    "val": {"pos_rmse_m": {"1": 0.01, "20": 0.2 / (i + 1) + offset}, "pos_mean_m": {"20": 0.1}, "att_rmse_deg": {"20": 1.0}, "reliable_horizon": 19.0},
                    "metric": 0.1, "best": 0.1, **({"resources": {"wall_s": 11.0, "cpu_s": 10.0, "cpu_pct": 91.0, "rss_mb": 900.0, "rss_peak_mb": 950.0,
                                                                  "gpu_peak_alloc_mb": 200.0, "gpu_peak_reserved_mb": 300.0}} if resources else {})})
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rec))
    (d / "provenance.json").write_text(json.dumps({"n_params": 1234, "git": "abc123def456", "torch": "2.x", "started": "2026-11-24 07:44:46"}))
    (d / "config.yaml").write_text(json.dumps({"train": {"horizon_stages": [1, 5, 10, 20], "stage_max_epochs": 8, "epochs": 60, "patience": 8, "val_horizon": 20,
                                                          "noise": "nominal", "val_noise": "nominal", "seed": 0, "batch_size": 256, "lr": 1e-3, "history": 10}}))
    if finished:
        (d / "summary.json").write_text(json.dumps({"epochs": 8, "best_metric": 0.2, "stop_reason": "epochs_done", "n_params": 1234, "total_wall_s": 3725.0,
                                                    "finished": "2026-11-24 08:46:51", "device": "cuda:1"}))


def _eval(models, offset):
    out = {}
    for m in models:
        out[m] = {}
        for role in ("val_id", "test_id", "test_ood_magnitude"):
            out[m][role] = {"n_windows": 100, "n_episodes": 5, "horizon": 20, "reliable_horizon_mean": 19.0,
                            "pos_rmse_m": {"1": 0.001, "5": 0.01, "10": 0.03, "20": 0.1 + offset * (m in R.USES_IMU)},
                            "pos_mean_m": {"20": 0.08}, "att_rmse_deg": {"20": 1.5}, "vel_rmse_mps": {"20": 0.2}, "omega_radps": {"20": 0.1}}
    return out


def test_report_builds_tables_figures_and_html(tmp_path):
    exp = tmp_path / "exp"
    for t, off in (("v05", 0.0), ("v06", 0.01)):
        for k in ("b2", "b3", "b4"):
            _run(exp, t, k, off, resources=(k == "b3"))
    _run(exp, "v06", "b5a", 0.0, finished=False)                                    # a run that is still going
    res = tmp_path / "res"; res.mkdir()
    with (res / "resource_samples.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_unix", "run", "dataset", "model", "pid", "gpu", "cpu_pct", "rss_mb", "rss_peak_mb", "gpu_mem_mb", "epoch", "horizon"])
        for i in range(30):
            for r, g in (("v05_b2", 0), ("v06_b2", 1), ("v06_b4", 2)):
                w.writerow([1795500000 + 2 * i, r, "d", r[4:], 100 + g, g, 100 + i % 5, 800 + i, 900 + i, 450 + g, i // 4 + 1, 5])
    with (res / "gpu_samples.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_unix", "gpu", "util_pct", "mem_used_mb"])
        for i in range(30):
            for g in range(4):
                w.writerow([1795500000 + 2 * i, g, 50 + g, 1000 + 100 * g])
    ev = tmp_path / "ev"; ev.mkdir()
    for t, off in (("v05", 0.0), ("v06", 0.02)):
        (ev / f"{t}.json").write_text(json.dumps(_eval(["b0", "b1", "b2", "b3", "b4"], off)))
    for a, b in (("v05", "v06"), ("v06", "v05")):
        (ev / f"{a}_on_{b}.json").write_text(json.dumps(_eval(["b2", "b3", "b4"], 0.05)))
    out = tmp_path / "out"
    html_path = R.build(exp, ["v05", "v06"], out, ev, res)
    assert html_path.exists() and (out / "REPORT.md").exists()
    figs = {p.name for p in (out / "figures").glob("*.png")}
    assert {"training_loss.png", "validation_error.png", "throughput_by_epoch.png", "best_validation_error.png", "training_time.png", "peak_gpu_memory.png",
            "peak_cpu_memory.png", "run_timeline_cpu.png", "gpu_utilisation_memory.png", "eval_pos_rmse_by_role.png", "eval_error_vs_step.png",
            "cross_evaluation.png"} <= figs
    rows = list(csv.DictReader((out / "tables" / "runs.csv").open()))
    assert len(rows) == 7 and {r["status"] for r in rows} == {"finished", "running"}
    b3 = next(r for r in rows if r["tag"] == "v06" and r["kind"] == "b3")
    assert float(b3["total_wall_s"]) == 3725.0 and float(b3["peak_gpu_mem_mb"]) == 300.0 and float(b3["peak_rss_mb"]) == 950.0       # per-epoch resource logging
    b2 = next(r for r in rows if r["tag"] == "v05" and r["kind"] == "b2")
    assert float(b2["peak_gpu_mem_mb"]) == 450.0 and float(b2["peak_rss_mb"]) == 929.0 and abs(float(b2["mean_cpu_pct"]) - 102.0) < 1.0     # external sampler
    assert abs(float(b2["best_val_pos_rmse_m"]) - 0.2 / 8) < 1e-9 and int(b2["best_epoch"]) == 8
    delta = list(csv.DictReader((out / "tables" / "v06_vs_v05_change.csv").open()))
    imu = [d for d in delta if d["uses_imu"] == "yes"]; ctl = [d for d in delta if d["uses_imu"] == "no"]
    assert imu and ctl and all(abs(float(d["change_pct"]) - 20.0) < 1e-6 for d in imu) and all(abs(float(d["change_pct"])) < 1e-9 for d in ctl)
    html_text = html_path.read_text()
    assert html_text.count("data:image/png;base64,") >= 10 and "<table>" in html_text and "Limitations" in html_text and "total_wall" not in html_text
    assert "1h 02m 05s" in html_text                                               # 3725 s formatted
    md = (out / "REPORT.md").read_text()
    assert "![" in md and "v0.5.0 (raw Isaac IMU)" in md and "one seed" in md


def test_report_degrades_gracefully_without_evaluation_or_resources(tmp_path):
    exp = tmp_path / "exp"
    for t in ("v05", "v06"):
        _run(exp, t, "b2", 0.0, finished=False)
    out = tmp_path / "out"
    R.build(exp, ["v05", "v06"], out, tmp_path / "none", None)
    assert (out / "report.html").exists() and not (out / "figures" / "eval_pos_rmse_by_role.png").exists()
    assert (out / "figures" / "training_loss.png").exists() and "Evaluation" not in (out / "REPORT.md").read_text()


def test_formatting_helpers():
    assert R.hms(3725) == "1h 02m 05s" and R.hms(None) == "–" and R.fmt(None) == "–" and R.fmt(0.123456, 3) == "0.123"
    assert R.md_table([{"a": 1.5, "b": None}], [("a", "A"), ("b", "B")]).splitlines()[0] == "| A | B |"


def test_analysis_sections_tables_and_figure(tmp_path):
    exp = tmp_path / "exp"
    for t in ("v05", "v06"):
        _run(exp, t, "b3", 0.0)
    ev = tmp_path / "ev"; ev.mkdir()
    for t, off in (("v05", 0.0), ("v06", 0.01)):
        def block(reg, n):
            return {"n_episodes": n, "n_windows": 10 * n, "pos_rmse_m": {m: {"1": 0.001, "5": 0.01, "20": 0.1 + off + (0.05 if m == "b1" else 0)} for m in ("b1", "b3", "b6a_future")},
                    "gaps": {"b3:b6a_future": {"ks": [1, 5, 20], "rmse_a": [0.1, 0.2, 0.3], "rmse_b": [0.1, 0.1, 0.1], "gap": [0.0, 0.1, 0.2],
                                               "gap_ci95": [[0.0, 0.05, 0.1], [0.0, 0.15, 0.3]], "n_windows": 10 * n, "n_episodes": n}}}
        (ev / f"analysis_regime_{t}.json").write_text(json.dumps({"ks": [1, 5, 20], "regimes": {"wind": block("wind", 8), "no_wind": block("no_wind", 2)}}))
        (ev / f"analysis_anchor_{t}.json").write_text(json.dumps({"mode": "h", "models": {"b3": {"n_windows": 50, "open_loop": {"5": 0.02, "20": 0.03}, "phi_vs_real_imu_boundary": {"5": 0.0, "20": 0.001},
                                                                                         "anchored": {"1": {"rmse": {"5": 0.02, "20": 0.029}}, "5": {"rmse": {"5": 0.02, "20": 0.03}}}}}}))
        (ev / f"analysis_residual_{t}.json").write_text(json.dumps({"models": {"b3": {"wind": {"n_windows": 40, "ratio_r_acc_to_f_spec": [0.03, 0.04], "ratio_r_ang_to_f_ang": [0.3, 0.2],
                                                                                              "degradation_ratio": {"1": 1.0, "20": 1.02}}}}}))
        (ev / f"analysis_shuffle_{t}.json").write_text(json.dumps({"ks": [1, 5, 20], "models": {"b1": {"n_windows": 30, "jump_ratio": {"1": 1.0, "5": 1.0, "20": 0.97}}}}))
    out = tmp_path / "out"
    R.build(exp, ["v05", "v06"], out, ev, None)
    assert (out / "figures" / "analysis_wind_vs_no_wind.png").exists()
    for name in ("regime", "gap", "anchor", "residual", "shuffle"):
        assert (out / "tables" / f"analysis_{name}.csv").exists(), name
    gap = list(csv.DictReader((out / "tables" / "analysis_gap.csv").open()))
    assert len(gap) == 4 and float(gap[0]["gap"]) == 0.2 and float(gap[0]["ci_high"]) == 0.3 and gap[0]["pair"] == "b3:b6a_future"
    md = (out / "REPORT.md").read_text()
    assert "## 5. Analyses" in md and "Shuffled-u" in md and "Paired gaps" in md and "## 6. Limitations" in md
