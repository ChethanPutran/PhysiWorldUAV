"""Provenance capture (plan §1-V1, §6.6-RP3).

Collects everything needed to reproduce or audit a dataset: resolved config
and its hash, git state of the project and of OmniDrones, simulator and
library versions, hardware, timestamps. Never raises for a missing optional
component — it records ``None`` with the reason instead.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ISAAC_ROOT = Path("/isaac-sim")


def config_hash(resolved_config: Any) -> str:
    """SHA-256 of the canonical JSON form of a resolved (plain) config."""
    text = json.dumps(resolved_config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state(repo: Path) -> dict[str, Any]:
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain")
    return {
        "path": str(repo),
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def _module_version(name: str) -> str | None:
    try:
        mod = __import__(name)
    except Exception:  # noqa: BLE001 - optional dependency probe
        return None
    return str(getattr(mod, "__version__", "unknown"))


def _isaac_versions() -> dict[str, Any]:
    out: dict[str, Any] = {"isaac_sim": None, "extensions": {}}
    vfile = ISAAC_ROOT / "VERSION"
    if vfile.exists():
        out["isaac_sim"] = vfile.read_text().strip()
    for ext in ("omni.isaac.core", "omni.isaac.sensor", "omni.physx", "omni.physx.tensors"):
        toml = ISAAC_ROOT / "exts" / ext / "config" / "extension.toml"
        if not toml.exists():
            toml = ISAAC_ROOT / "extsPhysics" / ext / "config" / "extension.toml"
        if toml.exists():
            for line in toml.read_text().splitlines():
                if line.strip().startswith("version"):
                    out["extensions"][ext] = line.split("=", 1)[1].strip().strip('"')
                    break
    return out


def _gpus() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [l.strip() for l in out.stdout.splitlines() if l.strip()] if out.returncode == 0 else []


def collect(resolved_config: Any | None = None, *, include_hardware: bool = True) -> dict[str, Any]:
    """Return the provenance record (JSON-serialisable)."""
    torch_cuda = None
    try:
        import torch
        torch_cuda = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    record: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_git": git_state(PROJECT_ROOT),
        "omnidrones_git": git_state(PROJECT_ROOT / "OmniDrones"),
        "python": platform.python_version(),
        "libraries": {
            name: _module_version(name)
            for name in ("numpy", "scipy", "torch", "h5py", "omegaconf", "hydra",
                         "tensordict", "torchrl", "omni_drones")
        },
        "torch_cuda": torch_cuda,
        "simulator": _isaac_versions(),
    }
    if include_hardware:
        record["host"] = socket.gethostname()
        record["gpus"] = _gpus()
    if resolved_config is not None:
        record["config_hash"] = config_hash(resolved_config)
    return record
