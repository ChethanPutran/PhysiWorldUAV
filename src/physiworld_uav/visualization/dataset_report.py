"""Self-contained HTML report of a generated (or planned) dataset.

Reads ``manifest.jsonl`` (the plan, available as soon as ``plan`` has run) and,
when present, ``index.json`` (written when generation finishes), and renders one
HTML file with: split roles, scenarios, parts, wind, vehicle-domain coverage,
counterfactual pairs and a filterable episode table. No server, no network, no
plotting library: the data is embedded in the page.

    python -m physiworld_uav.visualization.dataset_report --dataset data/datasets/physiworld_pilot_v0.5.0
"""
from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TEMPLATE = Path(__file__).parent / "templates" / "dataset_report.html"


def load_manifest(dataset_dir: Path) -> list[dict]:
    path = Path(dataset_dir) / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `generate plan` first")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_index(dataset_dir: Path) -> dict | None:
    path = Path(dataset_dir) / "index.json"
    return json.loads(path.read_text()) if path.exists() else None


def _counterfactual_kind(cf: dict | None) -> str:
    if not cf:
        return "none"
    if cf.get("is_baseline"):
        return "baseline"
    keys = list((cf.get("diff") or {}).keys())
    if any("wind_scale" in k for k in keys):
        return "wind-scale sibling"
    if any(k.startswith("domain") for k in keys):
        return "domain sibling"
    return "sibling"


def summarize_episode(r: dict, written: dict[str, str] | None = None) -> dict:
    """The compact per-episode record the page needs (a few dozen scalar fields)."""
    wind = r.get("wind") or {}
    stats = wind.get("stats") or {}
    labels = wind.get("labels") or {}
    dom = r.get("domain") or {}
    ref = r.get("reference") or {}
    return dict(
        id=r["episode_id"], part=r.get("part", "?"), grp=r.get("group_id", "?"), split=r.get("split_role", "?"),
        dt=r.get("data_type", "?"), scn=r.get("scenario_template", "?"),
        fam=ref.get("family") or "open-loop program", exc=r.get("excitation") or "—",
        eg=r.get("experiment_group") or "—", dom=dom.get("domain_id", "?"), reg=dom.get("region", "?"),
        ood=r.get("ood_category") or "in-distribution", wc=wind.get("category", "none"),
        wp=wind.get("profile", "none"), gust=labels.get("gust_class", "none"), tmp=labels.get("temporal_class", "none"),
        wm=round(float(stats.get("mean_horizontal_speed", 0.0)), 3), wpk=round(float(stats.get("peak_speed", 0.0)), 3),
        dur=r.get("duration_s", 0.0), n=r.get("n_steps", 0), ctrl=(r.get("controller") or {}).get("mode", "?"),
        cf=_counterfactual_kind(r.get("counterfactual")), replay=r.get("replay_of") is not None,
        f={k: round(float(v), 4) for k, v in (dom.get("factors") or {}).items()},
        mass=round(float((dom.get("drone") or {}).get("mass_kg", 0.0)), 4),
        written=(r["episode_id"] in written) if written is not None else None,
        status=(written or {}).get(r["episode_id"]),
    )


def _version_from_dir(dataset_dir: Path | None) -> str:
    m = re.search(r"_v(\d[\w.\-]*)$", Path(dataset_dir).name) if dataset_dir else None
    return m.group(1) if m else "?"


def build_meta(manifest: list[dict], index: dict | None, dataset_dir: Path | None = None,
               live: bool = False) -> dict:
    first = manifest[0] if manifest else {}
    v = (index or {}).get("validation") or {}
    return dict(
        dataset_id=(index or {}).get("dataset_id") or first.get("dataset_id", "dataset"),
        dataset_version=(index or {}).get("dataset_version") or _version_from_dir(dataset_dir),
        has_index=index is not None, live=live,
        n_planned=(index or {}).get("n_planned", len(manifest)),
        n_written=(index or {}).get("n_written", 0),
        complete=bool((index or {}).get("complete", False)),
        in_progress=bool((index or {}).get("in_progress", False)),
        n_failed=len((index or {}).get("failures") or []),
        validated=v.get("ok") is not None,
        validation_ok=bool(v.get("ok", False)), hard=v.get("hard") or 0, soft=v.get("soft") or 0,
    )


def _script_safe(obj) -> str:
    # keep the JSON from closing the <script> element
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def build_report(dataset_dir: str | Path, live: bool = False) -> str:
    """``live=True`` makes the page poll ``/index.json`` (only works when served with ``--serve``)."""
    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir)
    index = load_index(dataset_dir)
    written = {e["episode_id"]: e.get("status", "?") for e in index["episodes"]} if index and "episodes" in index else None
    rows = [summarize_episode(r, written) for r in manifest]
    return (TEMPLATE.read_text()
            .replace("__DATA__", _script_safe(rows))
            .replace("__META__", _script_safe(build_meta(manifest, index, dataset_dir, live))))


def _assign_workers(rows: list[dict], n_workers: int) -> dict[str, int]:
    """group key -> worker index; the same longest-processing-time assignment as ``experiments.generate.assign_workers`` (copied so that this
    module stays stdlib-only; a test keeps the two identical)."""
    key = lambda r: str(r.get("group_id") or r["episode_id"])  # noqa: E731
    cost: dict[str, float] = {}
    for r in rows:
        cost[key(r)] = cost.get(key(r), 0.0) + float(r.get("n_steps") or r.get("duration_s") or 1.0)
    load, out = [0.0] * n_workers, {}
    for k in sorted(cost, key=lambda k: (-cost[k], k)):
        w = min(range(n_workers), key=lambda i: (load[i], i))
        out[k] = w; load[w] += cost[k]
    return out


def live_workers(experiment_id: str) -> dict[int, int]:
    """{worker index: n_workers} of the generator processes of this experiment that are running on this machine (parsed from ``ps``)."""
    import subprocess
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return {}
    found: dict[int, int] = {}
    for line in out.splitlines():
        if "generate run" in line and "--shard" in line and f"experiment={experiment_id}" in line and not line.lstrip().startswith(("bash", "sh ", "/bin/bash")):
            try:
                i, n = line.split("--shard")[1].split()[0].split("/")
                found[int(i)] = int(n)
            except (IndexError, ValueError):
                continue
    return found


def live_states(manifest: list[dict], done: set[str], failed: set[str], alive: dict[int, int]) -> dict:
    """Per episode: done / failed (from the index files), running (the next unwritten episode of each live worker, in the order the worker
    processes them: manifest order with Mode B replays last), queued (assigned to a live worker, waiting) or pending (not assigned to a running
    worker). Also one record per worker."""
    n_workers = max(alive.values()) if alive else 0
    running, queued, workers = [], [], []
    if n_workers:
        owner = _assign_workers(manifest, n_workers)
        for w in range(n_workers):
            mine = [r for r in manifest if owner[str(r.get("group_id") or r["episode_id"])] == w]
            mine = sorted(mine, key=lambda r: r.get("replay_of") is not None)              # Mode B baselines first, as the generator does
            todo = [r["episode_id"] for r in mine if r["episode_id"] not in done and r["episode_id"] not in failed]
            is_alive = w in alive
            cur = todo[0] if (is_alive and todo) else None
            if cur:
                running.append(cur)
            if is_alive:
                queued += todo[1:]
            workers.append({"worker": w, "gpu": w % 3, "alive": is_alive, "assigned": len(mine), "done": len(mine) - len(todo), "queued": max(len(todo) - 1, 0) if is_alive else 0,
                            "current": cur})
    return {"running": running, "queued": queued, "workers": workers}


def combined_index(dataset_dir: Path) -> tuple[bytes, float] | None:
    """The live ``/index.json``: the merged ``index.json`` when it exists and is complete; otherwise (a parallel generation has no
    ``index.json`` until its merge) one assembled from the per-worker ``index.workerNN.json`` files. Returns (body, age in seconds)."""
    d = Path(dataset_dir)
    main = d / "index.json"
    if main.exists():
        try:
            idx = json.loads(main.read_text())
        except json.JSONDecodeError:
            idx = None
        if idx is not None and not idx.get("in_progress", False):
            return main.read_bytes(), max(0.0, time.time() - main.stat().st_mtime)
    files = sorted(d.glob("index.worker*.json"))
    if not files:
        if main.exists():
            return main.read_bytes(), max(0.0, time.time() - main.stat().st_mtime)
        prov = d / "provenance.json"                                          # planned, nothing written yet: an empty progress record (not a 404)
        if not (d / "manifest.jsonl").exists() or not prov.exists():
            return None
        man = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip()]
        idx = {"episodes": [], "failures": [], "n_written": 0, "n_planned": len(man), "in_progress": True, "complete": False, "generation_wall_s": 0.0,
               "dataset_id": d.name, "note": "planned; no episode written yet",
               **live_states(man, set(), set(), live_workers(man[0].get("experiment_id", "")) if man else {})}
        return json.dumps(idx).encode(), 0.0
    eps, fails, walls, running, newest = [], [], [], False, 0.0
    for f in files:
        try:
            w = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue                                                         # a worker is rewriting its file
        eps += [{"episode_id": e, "status": "completed"} if isinstance(e, str) else e for e in w.get("episodes", [])]
        fails += w.get("failures", []); walls.append(float(w.get("generation_wall_s") or 0.0)); running |= bool(w.get("in_progress", False))
        newest = max(newest, f.stat().st_mtime)
    prov = d / "provenance.json"
    elapsed = max(max(walls, default=0.0), time.time() - prov.stat().st_mtime if prov.exists() and running else 0.0)
    man = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip()]
    ids = {e["episode_id"] for e in eps}; bad = {f.get("episode_id") for f in fails}
    idx = {"episodes": eps, "failures": fails, "n_written": len(eps), "n_planned": len(man), "in_progress": running, "complete": False,
           "generation_wall_s": elapsed, "dataset_id": d.name, "note": "assembled from the per-worker index files (status of finished episodes not known yet)",
           **live_states(man, ids, bad, live_workers(man[0].get("experiment_id", "")) if man else {})}
    return json.dumps(idx).encode(), max(0.0, time.time() - newest)


def make_handler(dataset_dir: Path):
    """Serve exactly two paths: the report page (rebuilt per request) and the live index.json."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/report.html"):
                try:
                    self._send(200, build_report(dataset_dir, live=True).encode("utf-8"), "text/html; charset=utf-8")
                except Exception as exc:  # noqa: BLE001
                    self._send(500, f"could not build the report: {exc}".encode(), "text/plain; charset=utf-8")
            elif path == "/index.json":
                got = combined_index(dataset_dir)
                if got is not None:
                    self._send(200, got[0], "application/json", {"X-Index-Age-S": f"{got[1]:.1f}"})
                else:
                    self._send(404, b'{"missing":true}', "application/json")
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

    return Handler


def serve(dataset_dir: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(Path(dataset_dir)))
    print(f"serving {dataset_dir} at http://{host}:{server.server_address[1]}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> Path | None:
    p = argparse.ArgumentParser(prog="dataset-report", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True, help="dataset directory (contains manifest.jsonl)")
    p.add_argument("--out", type=Path, default=None, help="output HTML (default: <dataset>/report.html)")
    p.add_argument("--serve", action="store_true",
                   help="serve a live page that follows index.json while generation runs (no file is written)")
    p.add_argument("--host", default="127.0.0.1", help="bind address for --serve (default: local only)")
    p.add_argument("--port", type=int, default=8000, help="port for --serve; 0 picks a free one (this is a shared server)")
    a = p.parse_args(argv)
    if a.serve:
        serve(a.dataset, a.host, a.port)
        return None
    out = a.out or a.dataset / "report.html"
    out.write_text(build_report(a.dataset))
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
