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
import os
import re
import shutil
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


_PREV: dict[int, tuple[float, float]] = {}                      # pid -> (time, cpu ticks) of the previous sample, for the CPU %
_HIST: dict[int, list[tuple[float, int]]] = {}                  # worker -> [(time, episodes written)], for the per-worker rate
_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_GH: dict[str, list] = {}                                       # dataset dir -> [[time, episodes written], ...] for the "episodes over time" chart
_GH_SAVED: dict[str, float] = {}


def _smi(args: list[str]) -> list[list[str]]:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return []
    return [[c.strip() for c in l.split(",")] for l in out.strip().splitlines() if l.strip()]


def worker_pids(experiment_id: str) -> dict[int, int]:
    """{worker index: host pid} of the running generator processes of this experiment (the python process, not a wrapping shell)."""
    import subprocess
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return {}
    found: dict[int, int] = {}
    for line in out.splitlines():
        if "generate run" in line and "--shard" in line and f"experiment={experiment_id}" in line:
            parts = line.split(None, 2)
            if len(parts) < 3 or parts[1].endswith(("bash", "sh", "timeout")):
                continue
            try:
                found[int(line.split("--shard")[1].split()[0].split("/")[0])] = int(parts[0])
            except (IndexError, ValueError):
                continue
    return found


def _proc(pid: int) -> dict | None:
    try:
        st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        rss = 0.0
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS"):
                rss = float(line.split()[1]) / 1048576
        btime = next((float(l.split()[1]) for l in Path("/proc/stat").read_text().splitlines() if l.startswith("btime")), None)
        started = btime + float(st[19]) / _TICKS if btime else None                       # process start time (epoch seconds)
        return {"ticks": float(st[11]) + float(st[12]), "threads": int(st[17]), "rss_gb": rss, "started": started}
    except (OSError, IndexError, ValueError):
        return None


def _episode_progress(d: Path, w: int, pid: int | None, n_done: int, rate_per_h: float | None, D: int, now: float, started: float | None = None) -> dict:
    """What the worker is doing right now: the episode, the model step reached out of the planned steps, the time spent on it and the typical time.
    Exact when the worker writes ``progress.workerNN.json`` (generators started after 2026-11-26); otherwise estimated from when its index file was
    last written (= when it finished its previous episode) and the mean time per episode, flagged ``exact: false``."""
    out = {"ep_id": None, "ep_step": None, "ep_steps": None, "ep_elapsed_s": None, "ep_typical_s": None, "ep_pct": None, "ep_exact": False}
    pf, ix = d / f"progress.worker{w:02d}.json", d / f"index.worker{w:02d}.json"
    prog = None
    try:
        prog = json.loads(pf.read_text())
        if now - float(prog.get("updated", 0)) > 20 and pid:                  # stale (the worker is in start-up or between episodes): do not trust the step
            prog = prog if prog.get("step") == prog.get("n_steps") else None
    except (OSError, ValueError, json.JSONDecodeError):
        prog = None
    typical = (prog or {}).get("mean_episode_s") or (3600.0 / rate_per_h if rate_per_h else None)
    if typical is None and started and n_done:                                  # no recent rate yet: lifetime average (includes the start-up, so a little high)
        typical = max(now - started, 1.0) / n_done
    if prog and prog.get("step") is not None and pid and prog.get("step") < prog.get("n_steps", 0):
        out.update(ep_id=prog.get("episode_id"), ep_step=int(prog["step"] // D), ep_steps=int(prog["n_steps"] // D), ep_exact=True,
                   ep_elapsed_s=max(0.0, now - float(prog.get("t_start", now))), ep_typical_s=typical,
                   ep_pct=100.0 * prog["step"] / max(prog["n_steps"], 1))
        return out
    try:
        since = max(0.0, now - ix.stat().st_mtime) if n_done else None
    except OSError:
        since = None
    out.update(ep_elapsed_s=since, ep_typical_s=typical, ep_pct=min(99.0, 100.0 * since / typical) if since is not None and typical else None)
    return out


def resource_stats(dataset_dir: Path) -> dict:
    """CPU, memory, disk and GPU figures for the live page: host totals, our workers (per worker: pid, threads, RAM, CPU %, GPU memory,
    episodes written, shard bytes, episodes per hour) and the projected size of the finished dataset. Read from /proc, ``nvidia-smi`` and the
    dataset directory; no extra dependencies."""
    d = Path(dataset_dir); now = time.time()
    man_rows = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip()]
    total = len(man_rows); exp = man_rows[0].get("experiment_id", "") if man_rows else ""
    D = int(man_rows[0].get("sim", {}).get("control_decimation", 1)) if man_rows else 1
    # episodes and shard bytes per worker
    done: dict[int, int] = {}; wbytes: dict[int, int] = {}; allbytes = 0
    for f in d.glob("index.worker*.json"):
        try:
            w = int(f.stem.split("worker")[-1]); done[w] = len(json.loads(f.read_text()).get("episodes", []))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    for f in d.glob("shard_*.h5"):
        sz = f.stat().st_size; allbytes += sz
        if f.name.startswith("shard_w"):
            w = int(f.name[7:9]); wbytes[w] = wbytes.get(w, 0) + sz
    n_done = sum(done.values())
    hist_file = d / "report_history.json"                                  # survives a restart of the server
    gh = _GH.setdefault(str(d), [])
    if not gh and hist_file.exists():
        try:
            gh.extend(json.loads(hist_file.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    if not gh or gh[-1][1] != n_done or now - gh[-1][0] > 60:
        gh.append([now, n_done])
        del gh[:-3000]
        if now - _GH_SAVED.get(str(d), 0) > 30:
            try:
                hist_file.write_text(json.dumps(gh)); _GH_SAVED[str(d)] = now
            except OSError:
                pass
    # GPUs
    gpus = []
    for r in _smi(["--query-gpu=index,pci.bus_id,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"]):
        gpus.append({"index": int(r[0]), "bus": r[1].lower()[-12:], "util": float(r[2]), "mem_used_mb": float(r[3]), "mem_total_mb": float(r[4]),
                     "temp_c": float(r[5]) if r[5].replace(".", "").isdigit() else None, "power_w": float(r[6]) if r[6].replace(".", "").isdigit() else None})
    bus2idx = {g["bus"]: g["index"] for g in gpus}
    gpu_mem: dict[int, dict[int, float]] = {}                                  # pid -> {gpu index: MB}
    for r in _smi(["--query-compute-apps=pid,used_memory,gpu_bus_id"]):
        try:
            gpu_mem.setdefault(int(r[0]), {})[bus2idx.get(r[2].lower()[-12:], -1)] = float(r[1])
        except (ValueError, IndexError):
            continue
    # workers
    pids = worker_pids(exp); workers = []; our_cores = 0.0; our_ram = 0.0
    for w in sorted(set(done) | set(pids)):
        pid = pids.get(w); pr = _proc(pid) if pid else None
        cpu = None
        if pr:
            prev = _PREV.get(pid)
            if prev and now > prev[0] + 1:
                cpu = 100.0 * (pr["ticks"] - prev[1]) / _TICKS / (now - prev[0])
            _PREV[pid] = (now, pr["ticks"])
            our_ram += pr["rss_gb"]
            if cpu is not None:
                our_cores += cpu / 100
        h = _HIST.setdefault(w, []); n = done.get(w, 0)
        if not h or h[-1][1] != n or now - h[-1][0] > 120:
            h.append((now, n))
        del h[:-400]
        rec = [x for x in h if x[0] >= now - 900]
        rate = (rec[-1][1] - rec[0][1]) / max(rec[-1][0] - rec[0][0], 1) * 3600 if len(rec) > 1 and rec[-1][0] - rec[0][0] > 120 else None
        gm = gpu_mem.get(pid, {}) if pid else {}
        own = max(gm, key=gm.get) if gm else None                            # the GPU it computes on (it also holds small graphics contexts on the others)
        ep = _episode_progress(d, w, pid, n, rate, D, now, pr["started"] if pr else None)
        workers.append({**ep, "worker": w, "pid": pid, "alive": bool(pr), "threads": pr["threads"] if pr else None, "rss_gb": pr["rss_gb"] if pr else None,
                        "cpu_pct": cpu, "gpu": own, "gpu_mem_mb": gm.get(own) if own is not None else None, "gpu_mem_all_mb": sum(gm.values()) if gm else None,
                        "episodes": n, "disk_mb": wbytes.get(w, 0) / 1e6, "rate_per_h": rate})
    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1); meminfo[k] = float(v.split()[0]) / 1048576
    except OSError:
        pass
    other = sum(f.stat().st_size for f in d.iterdir() if f.is_file() and f.suffix != ".h5")
    per_ep = allbytes / n_done if n_done >= 10 else None
    du = shutil.disk_usage(d)
    return {"time": time.strftime("%H:%M:%S"), "dataset": d.name, "total": total, "written": n_done,
            "disk": {"shards_gb": allbytes / 1e9, "other_mb": other / 1e6, "per_episode_mb": per_ep / 1e6 if per_ep else None,
                     "projected_total_gb": (per_ep * total + other) / 1e9 if per_ep else None, "volume_free_gb": du.free / 1e9, "volume_total_gb": du.total / 1e9},
            "host": {"cores": os.cpu_count(), "load1": os.getloadavg()[0], "load5": os.getloadavg()[1], "load15": os.getloadavg()[2],
                     "ram_total_gb": meminfo.get("MemTotal"), "ram_available_gb": meminfo.get("MemAvailable")},
            "ours": {"workers_alive": sum(1 for x in workers if x["alive"]), "ram_gb": our_ram, "cpu_cores": our_cores if our_cores else None},
            "gpus": gpus, "workers": workers, "history": gh[-2000:]}


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
            elif path == "/stats.json":
                try:
                    self._send(200, json.dumps(resource_stats(dataset_dir)).encode(), "application/json")
                except Exception as exc:  # noqa: BLE001
                    self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")
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
