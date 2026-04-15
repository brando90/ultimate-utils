"""
job_scheduler_uu — Decentralized, file-based job scheduler (spool-directory pattern).

Designed for clusters that share a Distributed File System (DFS) but lack Slurm.
Multiple watcher daemons on different nodes poll a shared pending/ directory,
atomically claim jobs via os.link() + link-count check (NFS-safe), execute them,
and sort the results into completed/ or failed/.

Usage (daemon):
    python -m uutils.job_scheduler_uu.scheduler [--job-dir ~/dfs/job_queue] [--poll 15] [--timeout 14400]

Usage (submit a job):
    python -m uutils.job_scheduler_uu.submit my_script.sh [--job-dir ~/dfs/job_queue]
"""

import os

# Shared default so scheduler.py and submit.py don't duplicate this logic.
DEFAULT_JOB_DIR = os.path.join(os.path.expanduser("~"), "dfs", "job_queue")

__all__ = [
    "DEFAULT_JOB_DIR",
    "init_job_dirs",
    "claim_job",
    "execute_job",
    "watcher_loop",
    "submit_job",
]


def __getattr__(name: str):
    """Lazy imports to avoid RuntimeWarning when running submodules via -m."""
    if name in ("init_job_dirs", "claim_job", "execute_job", "watcher_loop"):
        from uutils.job_scheduler_uu.scheduler import (
            init_job_dirs,
            claim_job,
            execute_job,
            watcher_loop,
        )
        return locals()[name]
    if name == "submit_job":
        from uutils.job_scheduler_uu.submit import submit_job
        return submit_job
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
