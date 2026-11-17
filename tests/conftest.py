"""Shared pytest configuration.

* ``pytest``            → CPU-only tests (default)
* ``pytest --isaac``    → also Isaac Sim tests (inside the isaac-sim container)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def pytest_addoption(parser):
    parser.addoption("--isaac", action="store_true", default=False,
                     help="run tests that launch Isaac Sim")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--isaac"):
        return
    skip = pytest.mark.skip(reason="needs --isaac (Isaac Sim)")
    for item in items:
        if "isaac" in item.keywords:
            item.add_marker(skip)


def omni_drones_available() -> bool:
    try:
        import omni_drones.controllers  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def drone():
    from physiworld_uav.drones import load_drone
    return load_drone(ROOT / "config/drone/hummingbird.yaml")


@pytest.fixture(scope="session")
def root_dir() -> Path:
    return ROOT
