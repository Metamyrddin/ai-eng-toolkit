"""This file is read by pytest before any test runs. It sets two environment switches, tells Python
where the area folders are, and makes sure tests never overwrite the real results files. It
belongs to the tests folder.
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")  # must precede any triton import (environment_info() imports it)
os.environ.setdefault("AIETK_TEST_MODE", "1")  # common.env.save_json redirects writes under results/ to a temp dir
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for d in sorted(ROOT.glob("0[1-8]_*")):
    sys.path.insert(0, str(d))
