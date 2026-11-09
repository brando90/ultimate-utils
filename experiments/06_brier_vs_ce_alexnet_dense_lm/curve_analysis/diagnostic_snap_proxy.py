"""Add gateway SSH verbosity while executing the unchanged canonical SNAP guard.

This task-local wrapper accepts only the WikiText source host. It never reserves
or edits guard state itself, and never changes authentication or host-key flags.
"""

from pathlib import Path
import runpy
import subprocess
import sys


GUARD = Path.home() / "agents-config/scripts/snap_reconnect.py"
EXPECTED_CHILD = [
    "/usr/bin/ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectionAttempts=1",
    "-o", "ConnectTimeout=15", "-o", "ControlMaster=auto",
    "-o", "ControlPersist=8h", "-o", "ControlPath=~/.ssh/%r@%h:%p",
    "-W", "[mercury2.stanford.edu]:22", "brando9@whale.stanford.edu",
]


def diagnostic_args(command):
    if command == EXPECTED_CHILD:
        return [command[0], "-vv", *command[1:]]
    return command


def main():
    if sys.argv[1:] != ["mercury2.stanford.edu", "22"]:
        raise SystemExit("Diagnostic wrapper is restricted to mercury2.stanford.edu:22.")
    original_popen = subprocess.Popen

    def verbose_child(command, *args, **kwargs):
        return original_popen(diagnostic_args(command), *args, **kwargs)

    subprocess.Popen = verbose_child
    try:
        # Preserve canonical validation, admission, locking, deadlines, relay,
        # exception handling and exit codes, including a locally rejected read.
        runpy.run_path(str(GUARD), run_name="__main__")
    finally:
        subprocess.Popen = original_popen


if __name__ == "__main__":
    main()
