"""These tests check the shared helper code: seeding, saving JSON files and describing the machine.
"""
import json

import torch

from common import env


def test_seed_everything_is_reproducible():
    """Check that seeding twice gives the same random numbers."""
    env.seed_everything(123); a = torch.rand(5)
    env.seed_everything(123); b = torch.rand(5)
    assert torch.equal(a, b)


def test_save_json_roundtrip(tmp_path):
    """Check that a saved JSON file can be read back unchanged."""
    p = env.save_json({"a": 1, "b": [1, 2]}, tmp_path / "x" / "y.json")
    assert json.load(open(p)) == {"a": 1, "b": [1, 2]}


def test_environment_info_has_core_fields():
    """Check that the machine description contains the basic fields."""
    info = env.environment_info()
    assert info["python"] and info["torch"] and "cpu_count" in info
