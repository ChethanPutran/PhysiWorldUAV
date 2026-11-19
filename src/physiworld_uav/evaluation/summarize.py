"""Collect training runs and evaluation tables into one results folder.

    python -m physiworld_uav.evaluation.summarize [--experiments outputs/experiments] [--eval outputs/eval] [--out outputs/results]

Writes ``training_runs.csv`` (one row per epoch of every run), ``training_summary.md`` (one row per run),
``training_curves.png`` and copies the evaluation JSON/Markdown files, so the results of a gate run live in one place.
Safe to run while training is in progress (the summary says which runs are unfinished).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def load_runs(root: Path) -> dict[str, dict]:
    runs = {}
    for d in sorted(Path(root).iterdir()) if Path(root).exists() else []:
        mp = d / "metrics.jsonl"
        if not mp.exists():
            continue
        rec = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
        summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
        prov = json.loads((d / "provenance.json").read_text()) if (d / "provenance.json").exists() else {}
        cfg = json.loads((d / "config.yaml").read_text()) if (d / "config.yaml").exists() else {}
        runs[d.name] = dict(epochs=rec, summary=summ, provenance=prov, config=cfg)
    return runs


def write_csv(runs: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "epoch", "horizon", "lr", "train_total", "train_state", "train_roll", "val_pos1_m", "val_pos20_m", "val_att20_deg", "val_reliable_steps", "metric", "best", "epoch_s", "windows_per_s"])
        for name, r in runs.items():
            for e in r["epochs"]:
                v = e.get("val") or {}
                w.writerow([name, e["epoch"], e["horizon"], f"{e['lr']:.3g}", f"{e['train']['total']:.6g}", f"{e['train']['state']:.6g}", f"{e['train']['roll']:.6g}",
                            f"{v['pos_m']['1']:.5g}" if v else "", f"{v['pos_m']['20']:.5g}" if v and "20" in v["pos_m"] else "",
                            f"{v['att_deg']['20']:.4g}" if v and "20" in v["att_deg"] else "", f"{v['reliable_horizon']:.3g}" if v else "",
                            f"{e['metric']:.6g}", f"{e['best']:.6g}", f"{e['epoch_s']:.2f}", f"{e['windows_per_s']:.0f}"])


def markdown(runs: dict) -> str:
    L = ["# Training runs", "", "| run | model | status | epochs | horizon | first train loss | last train loss | loss ratio | best metric | last val pos@20 [m] | params | s/epoch |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        e0, e1 = r["epochs"][0], r["epochs"][-1]; v = e1.get("val")
        status = "finished" if r["summary"] else "running"
        v_pos = format(v["pos_m"]["20"], ".4f") if v and "20" in v["pos_m"] else "–"
        stop = f" ({r['summary']['stop_reason']})" if r["summary"] else ""
        ratio = e0["train"]["total"] / max(e1["train"]["total"], 1e-12)
        spe = sum(x["epoch_s"] for x in r["epochs"]) / len(r["epochs"])
        L.append(f"| {name} | {r['config'].get('train', {}).get('kind', '?')} | {status}{stop} | {len(r['epochs'])} | {e1['horizon']} | "
                 f"{e0['train']['total']:.4g} | {e1['train']['total']:.4g} | {ratio:.0f}× | {e1['best']:.4g} | {v_pos} | {r['provenance'].get('n_params', '?')} | {spe:.1f} |")
    L += ["", "Train loss is the thesis loss (decoder term + rescaled rollout term) on the current curriculum horizon; it is **not comparable across horizons**. "
          "Validation position error is on `val_id`, 20-step rollouts, given initial state."]
    return "\n".join(L)


def curves(runs: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for name, r in runs.items():
        ep = [e["epoch"] for e in r["epochs"]]
        ax[0].plot(ep, [e["train"]["total"] for e in r["epochs"]], label=name)
        v = [(e["epoch"], e["val"]["pos_m"]["20"]) for e in r["epochs"] if e.get("val") and "20" in e["val"]["pos_m"]]
        if v:
            ax[1].plot(*zip(*v), label=name)
    ax[0].set(title="training loss (log)", xlabel="epoch", yscale="log"); ax[1].set(title="val_id position error at 20 steps [m]", xlabel="epoch", yscale="log")
    for a in ax:
        a.grid(alpha=.25); a.legend(fontsize=7)
    fig.savefig(path, dpi=120); plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--experiments", type=Path, default=Path("outputs/experiments"))
    p.add_argument("--eval", type=Path, default=Path("outputs/eval"))
    p.add_argument("--out", type=Path, default=Path("outputs/results"))
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(a.experiments)
    write_csv(runs, a.out / "training_runs.csv")
    (a.out / "training_summary.md").write_text(markdown(runs))
    if runs:
        curves(runs, a.out / "training_curves.png")
    for f in Path(a.eval).glob("*") if Path(a.eval).exists() else []:
        shutil.copy2(f, a.out / f.name)
    (a.out / "runs.jsonl").write_text((a.experiments / "runs.jsonl").read_text()) if (a.experiments / "runs.jsonl").exists() else None
    print(f"wrote {a.out}: {sorted(x.name for x in a.out.iterdir())}")


if __name__ == "__main__":
    main()
