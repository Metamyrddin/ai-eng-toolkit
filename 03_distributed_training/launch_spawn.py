"""This file starts a script in several worker processes on the same computer, as a simpler stand-
in for the torchrun tool. It is part of area 3 and also makes the demos work on Windows.

More technical notes:
Launch a script in N local processes without torchrun (handy on Windows / for debugging).

    python launch_spawn.py ddp_cpu.py --world 2 [-- extra script args]
Sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT exactly like torchrun does.
"""
import argparse
import os
import socket
import subprocess
import sys


def free_port():
    """Ask the operating system for a network port that nobody is using right now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    """Start the chosen script in several worker processes on this computer, wait for all of
    them and return an error code if any of them failed.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--world", type=int, default=2)
    a, extra = ap.parse_known_args()
    extra = [e for e in extra if e != "--"]
    port = str(free_port())
    procs = []
    for r in range(a.world):
        env = dict(os.environ, RANK=str(r), LOCAL_RANK=str(r), WORLD_SIZE=str(a.world),
                   MASTER_ADDR="127.0.0.1", MASTER_PORT=port)
        procs.append(subprocess.Popen([sys.executable, a.script, *extra], env=env))
    codes = [p.wait() for p in procs]
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
