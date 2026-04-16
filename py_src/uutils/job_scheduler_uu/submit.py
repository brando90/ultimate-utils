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
import time
from pathlib import Path

from uutils.job_scheduler_uu import DEFAULT_JOB_DIR

_JOB_MODE_HEADER = "# JOB_MODE:"


def _validate_job_name(name: str) -> str:
    """Reject job names that would escape pending/ or corrupt watcher prompts."""
    if Path(name).name != name:
        raise ValueError(f"Job name must not contain path separators: {name!r}")
    if not name or name.strip(".") == "":
        raise ValueError(f"Invalid job name: {name!r}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise ValueError(f"Job name must not contain control characters: {name!r}")
    return name


def _deduplicate_dest(pending: Path, name: str) -> Path:
    """Return a non-colliding dot-prefixed staging Path under *pending*.

    If ``pending/name`` already exists, appends a timestamp.  If that still
    collides (two submissions in the same second), appends an incrementing
    counter until a free slot is found.

    Uses ``O_CREAT | O_EXCL`` on a dot-prefixed staging file to atomically
    claim the filename, preventing a TOCTOU race where two concurrent
    submitters could both pick the same destination.  The caller writes
    content to the staging file, then renames it to the final name via
    :func:`_finalize_dest`.  The dot prefix ensures the watcher daemon
    (which skips dotfiles) never executes a partially-written script.
    """
    stem = Path(name).stem
    suffix = Path(name).suffix
    ts = time.strftime("%Y%m%d_%H%M%S")

    # Build the list of candidate final names in priority order:
    # plain name -> timestamped -> counter-suffixed.
    def _candidates():
        yield name
        yield f"{stem}_{ts}{suffix}"
        counter = 1
        while counter <= 10000:
            yield f"{stem}_{ts}_{counter}{suffix}"
            counter += 1

    for final_name in _candidates():
        # Check that the final destination doesn't already exist.
        final_path = pending / final_name
        if final_path.exists():
            continue
        # Atomically create a dot-prefixed staging file.
        staging_path = pending / f".{final_name}"
        fd = _try_create_exclusive(staging_path)
        if fd is not None:
            os.close(fd)
            return staging_path
    raise RuntimeError(f"Could not find a free slot for {name!r} after 10000 attempts")


def _try_create_exclusive(path: Path) -> int | None:
    """Try to atomically create *path* with O_CREAT|O_EXCL.

    Returns the file descriptor on success, or None if the file already exists.
    """
    try:
        return os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return None


def _finalize_dest(staging: Path) -> Path:
    """Rename a dot-prefixed staging file to its final (non-dot) name.

    The watcher daemon skips dotfiles, so this rename is what makes the job
    visible to the scheduler.  ``os.rename()`` is atomic on a single
    filesystem (including NFS for same-directory renames).
    """
    final = staging.parent / staging.name.removeprefix(".")
    os.rename(str(staging), str(final))
    return final


def _inject_mode_header(staging: Path, mode_header: str) -> None:
    """Insert ``mode_header`` near the top of a staged text script."""
    if not mode_header:
        return
    content = staging.read_text(errors="replace")
    lines = content.splitlines(keepends=True)
    search_limit = min(len(lines), 20)

    for i in range(search_limit):
        if lines[i].strip().upper().startswith(_JOB_MODE_HEADER.upper()):
            lines[i] = mode_header
            staging.write_text("".join(lines))
            return

    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    lines.insert(insert_at, mode_header)
    content = "".join(lines)
    staging.write_text(content)


def submit_job(
    script: str | Path | None = None,
    *,
    inline: str | None = None,
    job_dir: str | Path = DEFAULT_JOB_DIR,
    job_name: str | None = None,
    mode: str | None = None,
) -> Path:
    """Copy *script* (or create one from *inline*) into pending/.

    The file is first written to a dot-prefixed staging path (invisible to
    the watcher daemon) and then atomically renamed to its final name.
    This prevents the watcher from claiming and executing a partially-written
    job script.

    If *mode* is given ("smart" or "direct"), a ``# JOB_MODE: <mode>`` header
    is injected into the script so the watcher knows how to execute it.
    If omitted, the watcher's ``--default-mode`` determines the mode.

    Returns the Path of the new file in pending/.
    """
    pending = Path(job_dir).expanduser().resolve() / "pending"
    pending.mkdir(parents=True, exist_ok=True)

    mode_header = f"# JOB_MODE: {mode}\n" if mode else ""

    if inline:
        # Generate a timestamped shell script from the inline command.
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = _validate_job_name(job_name or f"inline_{ts}.sh")
        # Use deduplication even for inline jobs — two inline submits in the
        # same second would otherwise overwrite each other.
        staging = _deduplicate_dest(pending, name)
        staging.write_text(f"#!/usr/bin/env bash\n{mode_header}set -euo pipefail\n{inline}\n")
        staging.chmod(staging.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        dest = _finalize_dest(staging)
    elif script:
        src = Path(script).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Script not found: {src}")
        name = _validate_job_name(job_name or src.name)
        staging = _deduplicate_dest(pending, name)
        shutil.copy2(str(src), str(staging))
        _inject_mode_header(staging, mode_header)
        # Ensure executable.
        staging.chmod(staging.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        dest = _finalize_dest(staging)
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
    parser.add_argument(
        "--mode",
        choices=["smart", "direct"],
        default=None,
        help="Execution mode. 'smart' = agent wraps job (diagnose failures, retry, "
             "email). 'direct' = plain subprocess. Omit to use watcher's --default-mode.",
    )
    args = parser.parse_args()

    if not args.script and not args.inline:
        parser.error("Provide a script path or --inline command.")

    submit_job(
        script=args.script,
        inline=args.inline,
        job_dir=args.job_dir,
        job_name=args.job_name,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
