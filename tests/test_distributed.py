"""These tests really start two processes and run the area 3 training scripts. Each script checks
its own results and fails loudly if they are wrong.

More technical notes:
Spawns real 2-process gloo jobs (CPU). Each script asserts exactness internally and exits non-zero on failure."""
import subprocess
import sys
from pathlib import Path

import pytest

D = Path(__file__).resolve().parent.parent / "03_distributed_training"


@pytest.mark.slow
@pytest.mark.parametrize("script", ["ddp_cpu.py", "fsdp_cpu.py", "collectives_demo.py"])
def test_two_process_jobs(script, tmp_path):
    """Run each distributed script on two processes and check that it exits without an error.
    """
    cmd = [sys.executable, str(D / "launch_spawn.py"), str(D / script), "--world", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=D)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
