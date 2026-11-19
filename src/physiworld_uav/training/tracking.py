"""Lightweight, dependency-free experiment tracking.

* every run appends events to ``<out_root>/runs.jsonl`` (start / epoch-summary / end) so runs can be listed and compared
  without opening their folders;
* an optional TensorBoard writer is used if ``tensorboard`` is importable (set ``use_tensorboard``);
* ``python -m physiworld_uav.training.tracking [--root outputs/experiments]`` prints a table of all runs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


class Tracker:
    def __init__(self, out_dir: Path, name: str, meta: dict, use_tensorboard: bool = False) -> None:
        self.out_dir, self.name = Path(out_dir), name
        self.index = self.out_dir.parent / "runs.jsonl"
        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(self.out_dir / "tb"))
            except Exception:  # noqa: BLE001 - optional dependency
                self.tb = None
        self._append({"event": "start", **meta})

    def _append(self, rec: dict) -> None:
        self.index.parent.mkdir(parents=True, exist_ok=True)
        with self.index.open("a") as f:
            f.write(json.dumps({"run": self.name, "t": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}) + "\n")

    def epoch(self, rec: dict) -> None:
        if self.tb is not None:
            self.tb.add_scalar("loss/train", rec["train"]["total"], rec["epoch"])
            self.tb.add_scalar("metric/val", rec["metric"], rec["epoch"])
            self.tb.add_scalar("lr", rec["lr"], rec["epoch"])

    def end(self, summary: dict) -> None:
        self._append({"event": "end", **summary})
        if self.tb is not None:
            self.tb.close()


def load_runs(root: Path) -> list[dict]:
    path = Path(root) / "runs.jsonl"
    runs: dict[str, dict] = {}
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        cur = runs.setdefault(r["run"], {"run": r["run"], "status": "running"})
        if r["event"] == "start":
            cur.update({k: v for k, v in r.items() if k != "event"}); cur["status"] = "running"
        elif r["event"] == "end":
            cur.update({k: v for k, v in r.items() if k != "event"}); cur["status"] = "finished"
    return list(runs.values())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=Path("outputs/experiments"))
    a = p.parse_args(argv)
    rows = load_runs(a.root)
    print(f"{'run':24s} {'model':6s} {'dataset':26s} {'epochs':>6s} {'best metric':>12s} {'params':>8s} {'status':9s} started")
    for r in rows:
        print(f"{r['run'][:24]:24s} {str(r.get('model','?')):6s} {str(r.get('dataset','?'))[:26]:26s} {str(r.get('epochs','-')):>6s} "
              f"{r.get('best_metric', float('nan')):12.4g} {str(r.get('n_params','-')):>8s} {r['status']:9s} {r.get('t','')}")


if __name__ == "__main__":
    main()
