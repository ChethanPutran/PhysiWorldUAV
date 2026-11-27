"""Push notifications to a phone through ntfy (https://ntfy.sh, or a self-hosted server). Standard library only; never raises.

Setup (once):  scripts/setup_ntfy.sh   -> creates a random topic in logs/ntfy.conf and prints how to subscribe in the ntfy app.
Use:           python -m physiworld_uav.notify "message" [--title T] [--priority high] [--tags white_check_mark]
               python -m physiworld_uav.notify --test
               from physiworld_uav.notify import notify;  notify("Generation finished: 4530/4530 episodes", title="Data generation")

Configuration (first found): environment NTFY_TOPIC / NTFY_SERVER / NTFY_TOKEN, else the file ``logs/ntfy.conf`` (lines ``topic=...``,
``server=...``, ``token=...``; the file is git-ignored). Without a topic every call does nothing.

PRIVACY: the topic is the only secret on the public server (anyone who knows it can read and post); the text is stored on the server for a
short time. Messages therefore carry no project, dataset, user or host names, only generic status text and numbers.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_SERVER = "https://ntfy.sh"


def _conf_file() -> Path:
    for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
        f = base / "logs" / "ntfy.conf"
        if f.exists():
            return f
    return Path(__file__).resolve().parents[2] / "logs" / "ntfy.conf"


def config() -> dict:
    """{"topic", "server", "token"}: environment first, then logs/ntfy.conf. ``topic`` is empty when notifications are not set up."""
    cfg = {"topic": os.environ.get("NTFY_TOPIC", ""), "server": os.environ.get("NTFY_SERVER", ""), "token": os.environ.get("NTFY_TOKEN", "")}
    f = _conf_file()
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() in cfg and not cfg[k.strip()]:
                    cfg[k.strip()] = v.strip()
    cfg["server"] = (cfg["server"] or DEFAULT_SERVER).rstrip("/")
    return cfg


def notify(message: str, title: str | None = None, priority: str = "default", tags: str | None = None, timeout: float = 10.0) -> bool:
    """Send ``message`` to the configured topic. True if the server accepted it; False if not set up or on any error (never raises)."""
    try:
        cfg = config()
        if not cfg["topic"]:
            return False
        headers = {"Priority": priority}
        if title:
            headers["Title"] = title.encode("ascii", "replace").decode()
        if tags:
            headers["Tags"] = tags
        if cfg["token"]:
            headers["Authorization"] = "Bearer " + cfg["token"]
        req = urllib.request.Request(f"{cfg['server']}/{cfg['topic']}", data=message.encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:  # noqa: BLE001
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("message", nargs="?", default="")
    ap.add_argument("--title"); ap.add_argument("--priority", default="default", choices=["min", "low", "default", "high", "urgent"])
    ap.add_argument("--tags"); ap.add_argument("--test", action="store_true", help="send a test message")
    a = ap.parse_args(argv)
    if not config()["topic"]:
        print("ntfy is not set up: run scripts/setup_ntfy.sh (or set NTFY_TOPIC)", file=sys.stderr)
        return 2
    msg = "ntfy test: this phone will receive the training and generation notices" if a.test else a.message
    if not msg:
        ap.error("a message or --test is required")
    ok = notify(msg, title=a.title or ("Test" if a.test else None), priority=a.priority, tags=a.tags or ("bell" if a.test else None))
    print("sent" if ok else "NOT sent (no connection or rejected)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
