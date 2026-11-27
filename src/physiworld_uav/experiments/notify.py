"""Watch a running dataset generation and send short status messages to a webhook.

Reads ``index.json`` (rewritten after every episode by ``generate run``) and posts a message when the run
starts, at every 10 % of progress, when an episode fails, when the index stops updating (stalled or dead
process), at a periodic heartbeat, and when it finishes. Nothing is sent unless a URL is configured:

    export PHYSIWORLD_NOTIFY_URL='https://ntfy.sh/<your-secret-topic>'     # name only here; the value is yours
    python -m physiworld_uav.experiments.notify --dataset data/datasets/<name> --format text

Messages contain only the dataset id, counts and episode ids; no paths, host names or user names.
Formats: ``text`` (ntfy: plain POST body), ``slack`` / ``discord`` / ``telegram`` (JSON; Telegram's chat id
goes in the URL query). Without a URL the messages are printed (dry run).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

URL_ENV = "PHYSIWORLD_NOTIFY_URL"


@dataclass
class WatchState:
    started: bool = False
    last_bucket: int = 0
    failures_seen: int = 0
    stalled: bool = False
    finished: bool = False
    last_heartbeat: float = field(default_factory=time.time)


def _fmt(sec: float) -> str:
    sec = max(0, int(sec))
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    m = r // 60
    return (f"{d}d " if d else "") + (f"{h}h " if h or d else "") + f"{m}m"


def _summary(idx: dict) -> str:
    n, total = idx["n_written"], idx["n_planned"]
    fails = len(idx.get("failures") or [])
    early = sum(1 for e in idx["episodes"] if e.get("status") != "completed")
    wall = idx.get("generation_wall_s") or 0.0
    eta = ""
    if idx.get("in_progress") and n and n < total:
        eta = f", about {_fmt(wall / n * (total - n - fails))} left"
    return (f"{idx['dataset_id']} v{idx['dataset_version']}: {n}/{total} written ({100 * n // max(1, total)} %), "
            f"{fails} failed, {early} ended early{eta}")


def decide(state: WatchState, idx: dict | None, age_s: float, *, now: float, stall_s: float = 900.0,
           heartbeat_s: float = 86400.0) -> list[str]:
    """Pure function: update ``state`` and return the messages to send for this observation."""
    if idx is None:
        return []
    out: list[str] = []
    total = max(1, idx["n_planned"])
    fails = len(idx.get("failures") or [])
    if idx.get("in_progress"):
        if not state.started:
            state.started = True
            out.append("Generation running. " + _summary(idx))
        bucket = 10 * (100 * idx["n_written"] // total // 10)
        if bucket > state.last_bucket:
            state.last_bucket = bucket
            out.append("Progress. " + _summary(idx))
        if age_s > stall_s and not state.stalled:
            state.stalled = True
            out.append(f"No update for {_fmt(age_s)}: the run may have stopped. " + _summary(idx))
        elif age_s <= stall_s and state.stalled:
            state.stalled = False
            out.append("Updates resumed. " + _summary(idx))
        if now - state.last_heartbeat >= heartbeat_s:
            state.last_heartbeat = now
            out.append("Still running. " + _summary(idx))
    if fails > state.failures_seen:
        ids = ", ".join(f["episode_id"] for f in (idx.get("failures") or [])[state.failures_seen:][:3])
        out.append(f"{fails - state.failures_seen} episode(s) failed ({ids}). " + _summary(idx))
        state.failures_seen = fails
    if not idx.get("in_progress") and not state.finished:
        state.finished = True
        v = idx.get("validation") or {}
        verdict = "complete" if idx.get("complete") else "finished with problems"
        vtxt = "validation passed" if v.get("ok") else f"validation: {v.get('hard')} hard / {v.get('soft')} soft findings"
        out.append(f"Generation {verdict}; {vtxt}. " + _summary(idx))
    return out


def send(url: str | None, text: str, fmt: str = "text") -> None:
    if not url:
        print(f"[dry run] {text}", flush=True)
        return
    if fmt == "text":
        data, ctype = text.encode("utf-8"), "text/plain; charset=utf-8"
    else:
        key = "content" if fmt == "discord" else "text"
        data, ctype = json.dumps({key: text}).encode("utf-8"), "application/json"
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype}, method="POST")
    urllib.request.urlopen(req, timeout=15).read()


def read_index(dataset: Path) -> tuple[dict | None, float]:
    f = Path(dataset) / "index.json"
    if not f.exists():
        return None, 0.0
    return json.loads(f.read_text()), max(0.0, time.time() - f.stat().st_mtime)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="notify", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--format", choices=["text", "slack", "discord", "telegram"], default="text")
    p.add_argument("--interval", type=float, default=60.0, help="seconds between checks")
    p.add_argument("--stall-s", type=float, default=900.0, help="alert if index.json is older than this while running")
    p.add_argument("--heartbeat-h", type=float, default=24.0, help="'still running' message period; 0 disables")
    a = p.parse_args(argv)
    url = os.environ.get(URL_ENV)
    state = WatchState()
    print(f"watching {a.dataset} every {a.interval:.0f} s; "
          f"{'sending to $' + URL_ENV if url else 'no URL set: dry run'}", flush=True)
    while True:
        try:
            idx, age = read_index(a.dataset)
            for msg in decide(state, idx, age, now=time.time(), stall_s=a.stall_s,
                              heartbeat_s=a.heartbeat_h * 3600 if a.heartbeat_h else float("inf")):
                try:
                    send(url, msg, a.format)
                except Exception as exc:  # noqa: BLE001  (a notification failure must not stop the watcher)
                    print(f"could not send: {type(exc).__name__}", flush=True)
            if state.finished:
                return
        except Exception as exc:  # noqa: BLE001
            print(f"watch error: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
