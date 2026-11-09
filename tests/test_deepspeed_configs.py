"""These tests check that the DeepSpeed configuration files of area 3 are valid, and that DeepSpeed
itself accepts them when it is installed.
"""
import glob
import json
import sys
from pathlib import Path

import pytest

CFG = sorted(glob.glob(str(Path(__file__).resolve().parent.parent / "03_distributed_training" / "deepspeed_configs" / "*.json")))


def test_files_exist_and_valid_json():
    """Check that the configuration files exist and contain valid JSON."""
    assert len(CFG) >= 4
    for f in CFG:
        json.load(open(f))


def _materialize_auto(obj):
    """Replace the placeholder 'auto' by a number, because the Hugging Face trainer would
    normally fill it in.
    """
    if isinstance(obj, dict):
        return {k: _materialize_auto(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_materialize_auto(v) for v in obj]
    return {"auto": 1}.get(obj, obj) if obj == "auto" else obj


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="not supported on native Windows; use WSL2",
)
@pytest.mark.parametrize("path", CFG)
def test_parsed_by_deepspeed(path):
    """Check that DeepSpeed accepts each configuration file. The test is skipped if DeepSpeed
    is not installed. On native Windows it is skipped because DeepSpeed has no normal installer.
    The checks that the files are valid JSON still run on Windows.
    """
    pytest.importorskip("deepspeed")
    from deepspeed.runtime.config import DeepSpeedConfig
    cfg = json.load(open(path))
    if "auto" in json.dumps(cfg):  # HF-Trainer templates: 'auto' is resolved by transformers; substitute plausible values
        cfg = json.loads(json.dumps(cfg).replace('"auto"', "1"))
        cfg["train_batch_size"] = cfg["train_micro_batch_size_per_gpu"] * cfg["gradient_accumulation_steps"]
        cfg["optimizer"]["params"].update({"betas": [0.9, 0.999], "lr": 1e-5, "eps": 1e-8, "weight_decay": 0.0})
        cfg["scheduler"]["params"].update({"total_num_steps": 100, "warmup_num_steps": 10, "warmup_min_lr": 0.0, "warmup_max_lr": 1e-5})
        cfg["gradient_clipping"] = 1.0
        cfg["bf16"]["enabled"] = True
    ds = DeepSpeedConfig(cfg)
    assert ds.zero_optimization_stage in (2, 3)
    assert ds.train_batch_size == ds.train_micro_batch_size_per_gpu * ds.gradient_accumulation_steps  # world size 1 here
