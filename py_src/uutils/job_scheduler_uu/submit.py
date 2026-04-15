#!/usr/bin/env python3
"""
submit.py — Helper to submit jobs to the DFS job queue.

Usage:
    python -m uutils.job_scheduler_uu.submit my_script.sh
    python -m uutils.job_scheduler_uu.submit my_script.sh --job-dir ~/dfs/job_queue
    python -m uutils.job_scheduler_uu.submit --inline "echo hello && nvidia-smi"
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
import time
from pathlib import Path

from uutils.job_scheduler_uu import DEFAULT_JOB_DIR


def _deduplicate_dest(pending: Path, name: str) -> Path:
    """Return a non-colliding Path under *pending* for the given *name*.

    If ``pending/name`` already exists, appends a timestamp.  If that still
    collides (two submissions in the same second), appends an incrementing
    counter until a free slot is found.

    Uses ``O_CREAT | O_EXCL`` to atomically claim the filename, preventing
    a TOCTOU race where two concurrent submitters could both pick the same
    destination and silently overwrite each other.
    """
    candidates = [pending / name]
    stem = Path(name).stem
    suffix = Path(name).suffix
    ts = time.strftime("%Y%m%d_%H%M%S")
    # We try the plain name first, then timestamped, then counter-suffixed.
    # _try_create_exclusive handles the atomic check-and-create.
    for candidate in candidates:
        fd = _try_create_exclusive(candidate)
        if fd is not None:
            os.close(fd)
            return candidate
    # Plain name was taken; try timestamped.
    candidate = pending / f"{stem}_{ts}{suffix}"
    fd = _try_create_exclusive(candidate)
    if fd is not None:
        os.close(fd)
        return candidate
    # Timestamped also taken; append counter.
    counter = 1
    while True:
        candidate = pending / f"{stem}_{ts}_{counter}{suffix}"
        fd = _try_create_exclusive(candidate)
        if fd is not None:
            os.close(fd)
            return candidate
        counter += 1
        if counter > 10000:
            raise RuntimeError(f"Could not find a free slot for {name!r} after 10000 attempts")


def _try_create_exclusive(path: Path) -> int | None:
    """Try to atomically create *path* with O_CREAT|O_EXCL.

    Returns the file descriptor on success, or None if the file already exists.
    """
    try:
        return os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return None


def submit_job(
    script: str | Path | None = None,
    *,
    inline: str | None = None,
    job_dir: str | Path = DEFAULT_JOB_DIR,
    job_name: str | None = None,
) -> Path:
    """Copy *script* (or create one from *inline*) into pending/.

    Returns the Path of the new file in pending/.
    """
    pending = Path(job_dir).expanduser().resolve() / "pending"
    pending.mkdir(parents=True, exist_ok=True)

    if inline:
        # Generate a timestamped shell script from the inline command.
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = job_name or f"inline_{ts}.sh"
        # Use deduplication even for inline jobs — two inline submits in the
        # same second would otherwise overwrite each other.
        dest = _deduplicate_dest(pending, name)
        dest.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{inline}\n")
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    elif script:
        src = Path(script).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Script not found: {src}")
        name = job_name or src.name
        dest = _deduplicate_dest(pending, name)
        shutil.copy2(str(src), str(dest))
        # Ensure executable.
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    else:
        raise ValueError("Provide either 'script' path or 'inline' command string.")

    print(f"Submitted: {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a job to the DFS job queue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python -m uutils.job_scheduler_uu.submit train.sh
    python -m uutils.job_scheduler_uu.submit --inline "echo hello && sleep 5"
    python -m uutils.job_scheduler_uu.submit train.sh --name my_experiment.sh
""",
    )
    parser.add_argument("script", nargs="?", help="Path to a .sh or .py script to submit")
    parser.add_argument("--inline", help="Inline shell command to wrap in a script and submit")
    parser.add_argument("--job-dir", default=DEFAULT_JOB_DIR, help="Root of the job queue directory")
    parser.add_argument("--name", dest="job_name", help="Override the job filename in pending/")
    args = parser.parse_args()

    if not args.script and not args.inline:
        parser.error("Provide a script path or --inline command.")

    submit_job(
        script=args.script,
        inline=args.inline,
        job_dir=args.job_dir,
        job_name=args.job_name,
    )


if __name__ == "__main__":
    main()
