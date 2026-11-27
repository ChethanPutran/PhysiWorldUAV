"""YAML configuration for training / model / evaluation (config/training, config/model, config/evaluation)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from physiworld_uav.training import run as train_cli
from physiworld_uav.training.config_io import build_dataclass, eval_defaults, merged, parse_override, train_config
from physiworld_uav.training.trainer import TrainConfig

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "config"
KINDS = {"b2", "b3", "b4", "b4_rw", "b5a", "b5b", "b6b_future", "b6b_current"}


def test_default_yaml_equals_the_dataclass_defaults():
    """The YAML defaults must not drift from TrainConfig: every field present, same value."""
    d = merged([CFG / "training" / "default.yaml"])
    assert set(d) == {f.name for f in dataclasses.fields(TrainConfig)}
    assert train_config([CFG / "training" / "default.yaml"]) == TrainConfig()


def test_every_model_and_training_yaml_builds_a_valid_config():
    for f in sorted((CFG / "model").glob("*.yaml")):
        c = train_config([CFG / "training" / "default.yaml", f])
        assert c.kind in KINDS, f.name
    for f in sorted((CFG / "training").glob("*.yaml")):
        train_config([CFG / "training" / "default.yaml", f])
    assert train_config([CFG / "training" / "default.yaml", CFG / "model" / "b3_enc_u.yaml"]).enc_uses_u is True
    assert train_config([CFG / "training" / "smoke.yaml"]).horizon_stages == [5, 10]


def test_precedence_unknown_keys_and_override_parsing():
    c = train_config([CFG / "training" / "default.yaml", CFG / "model" / "b3.yaml"], ["epochs=5", "noise=none", "horizon_stages=[1,5]", "lr=3e-4"])
    assert (c.kind, c.epochs, c.noise, c.horizon_stages, c.lr) == ("b3", 5, "none", [1, 5], 3e-4)
    assert parse_override("a=true") == ("a", True) and parse_override("a=x") == ("a", "x")
    with pytest.raises(ValueError, match="unknown TrainConfig key"):
        build_dataclass(TrainConfig, {"epohcs": 3})                  # a typo must not silently fall back to the default


def test_training_cli_merges_yaml_and_overrides(monkeypatch):
    seen = {}
    monkeypatch.setattr(train_cli, "train", lambda ds, out, cfg, **kw: seen.update(cfg=cfg, out=out) or {"epochs": 0})
    train_cli.main(["--dataset", "d", "--name", "n", "--config", str(CFG / "training/smoke.yaml"), str(CFG / "model/b3.yaml"), "--set", "epochs=3"])
    c = seen["cfg"]
    assert (c.kind, c.epochs, c.horizon_stages, c.val_horizon) == ("b3", 3, [5, 10], 20)          # model YAML, --set over YAML, smoke YAML
    train_cli.main(["--dataset", "d", "--name", "n", "--config", str(CFG / "model/b3.yaml"), "--model", "b2"])
    assert seen["cfg"].kind == "b2"                                    # explicit --model wins over the YAML
    train_cli.main(["--dataset", "d", "--name", "n"])
    assert seen["cfg"].kind == "b2" and seen["cfg"] == TrainConfig(kind="b2", device="auto")
    train_cli.main(["--dataset", "d", "--name", "n", "--tiny", "8", "--device", "cpu"])
    assert seen["cfg"].tiny_episodes == 8 and seen["cfg"].device == "cpu"
    with pytest.raises(SystemExit):
        train_cli.main(["--dataset", "d", "--name", "n", "--set", "bogus=1"])


def test_evaluation_defaults_file():
    d = eval_defaults(CFG / "evaluation" / "default.yaml")
    assert d["noise"] == "nominal" and d["horizon"] == 20 and "val_id" in d["roles"]
    assert eval_defaults(None) == {}
    bad = Path(__file__).parent / "_bad_eval.yaml"
    bad.write_text("horizon: 5\nnois: none\n")
    try:
        with pytest.raises(ValueError, match="unknown evaluation key"):
            eval_defaults(bad)
    finally:
        bad.unlink()
