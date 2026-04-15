#!/usr/bin/env python3
"""
scheduler.py — Decentralized DFS job-queue watcher daemon.

Architecture (spool-directory pattern):
    $HOME/dfs/job_queue/
        pending/      drop .sh or .json job files here
        running/      jobs claimed by a node (hardlinked with hostname)
        completed/    exit-code 0 jobs
        failed/       timed-out or non-zero exit-code jobs
        logs/         per-job stdout+stderr logs

Multiple instances on different servers poll pending/, race to claim files via
os.link() + link-count check (NFS-safe atomic claim), execute them, and sort
outcomes.  No central coordinator needed.

Why os.link() instead of os.rename():
    os.rename() is atomic on local POSIX filesystems but NOT reliably atomic
    across NFS clients — two nodes can both believe they won.  os.link()
    (hardlink creation) IS atomic on NFS v3+.  The protocol:
        1. link(pending/job.sh, running/job.sh.<hostname>)
        2. stat(pending/job.sh).st_nlink == 2 → we won
        3. unlink(pending/job.sh)
    If another node wins, our link() raises or the nlink check fails.

Run:
    python -m uutils.job_scheduler_uu.scheduler            # defaults
    python -m uutils.job_scheduler_uu.scheduler --poll 10   # faster polling
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from uutils.job_scheduler_uu import DEFAULT_JOB_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_POLL_SECONDS = 15
DEFAULT_TIMEOUT_SECONDS = 4 * 3600  # 4 hours
HOSTNAME = socket.gethostname()

# Separator between the original filename and the claiming hostname.
# Using a triple-underscore makes it far less likely to collide with
# underscores that naturally appear in job filenames or hostnames.
_CLAIM_SEP = "___"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [{HOSTNAME}] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directory initialisation
# ---------------------------------------------------------------------------

SUBDIRS = ("pending", "running", "completed", "failed", "logs")


def init_job_dirs(job_dir: str | Path) -> dict[str, Path]:
    """Create the spool directory tree if it does not already exist.

    Returns a dict mapping logical name -> absolute Path, e.g.
        {"pending": Path(".../pending"), "running": ..., ...}
    """
    job_dir = Path(job_dir).expanduser().resolve()
    dirs: dict[str, Path] = {}
    for name in SUBDIRS:
        p = job_dir / name
        p.mkdir(parents=True, exist_ok=True)
        dirs[name] = p
    log.info("Job directories initialised under %s", job_dir)
    return dirs


# ---------------------------------------------------------------------------
# Filename sanitisation
# ---------------------------------------------------------------------------


def _sanitize_filename(name: str) -> str:
    """Strip path separators and dangerous components from a job filename.

    Prevents path-traversal attacks (e.g. ``../../etc/cron.d/evil.sh``) by
    keeping only the basename and rejecting names that are empty or consist
    solely of dots.
    """
    # Take only the final path component — removes any directory traversal.
    name = os.path.basename(name)
    # Reject empty or dot-only names (e.g. ".", "..").
    if not name or name.strip(".") == "":
        raise ValueError(f"Invalid job filename after sanitisation: {name!r}")
    return name


# ---------------------------------------------------------------------------
# Atomic claim (NFS-safe via hardlink + nlink check)
# ---------------------------------------------------------------------------


def claim_job(pending_path: Path, running_dir: Path) -> Optional[Path]:
    """Attempt to atomically claim *pending_path* using NFS-safe hardlink.

    Protocol:
        1. os.link(pending/job.sh, running/job.sh___<hostname>)
        2. Check os.stat(pending/job.sh).st_nlink == 2  (we won the race)
        3. os.unlink(pending/job.sh)  (remove from pending)

    If another node created its hardlink first, either our link() raises
    FileExistsError/OSError, or the nlink count will be > 2 (multiple
    claimants).  In either failure case we clean up our link and return None.

    The destination filename uses ``___`` (triple underscore) as separator
    between the original name and the hostname, avoiding ambiguity with
    single underscores in filenames or hostnames.

    Returns the new Path inside running/ on success, or None if the file was
    already grabbed by another node.
    """
    safe_name = _sanitize_filename(pending_path.name)
    dest_name = f"{safe_name}{_CLAIM_SEP}{HOSTNAME}"
    dest = running_dir / dest_name

    try:
        # Step 1: Create a hardlink.  Atomic on NFS v3+.
        os.link(str(pending_path), str(dest))
    except (FileNotFoundError, FileExistsError, OSError) as exc:
        # Another node got it first, or file already gone.
        log.debug("Could not link %s (likely taken): %s", pending_path.name, exc)
        return None

    try:
        # Step 2: Check that the pending file now has exactly 2 links
        # (the original + our link).  If it's > 2, multiple nodes linked
        # simultaneously and we must back off.
        nlink = os.stat(str(pending_path)).st_nlink
        if nlink != 2:
            log.debug(
                "Lost race for %s (nlink=%d, expected 2) — removing our link",
                pending_path.name,
                nlink,
            )
            os.unlink(str(dest))
            return None
    except FileNotFoundError:
        # The pending file was already unlinked by the winner.  Check if our
        # dest still exists — if so, we might actually be the winner who just
        # hasn't unlinked yet.  But to be safe, treat this as a loss.
        try:
            os.unlink(str(dest))
        except FileNotFoundError:
            pass
        return None
    except OSError as exc:
        log.debug("Stat failed for %s during claim: %s", pending_path.name, exc)
        try:
            os.unlink(str(dest))
        except FileNotFoundError:
            pass
        return None

    # Step 3: We won.  Remove the original from pending/.
    try:
        os.unlink(str(pending_path))
    except FileNotFoundError:
        pass  # Another winner already removed it — fine, we have our copy.

    log.info("Claimed job: %s -> %s", pending_path.name, dest.name)
    return dest


# ---------------------------------------------------------------------------
# Original filename recovery
# ---------------------------------------------------------------------------


def _recover_original_name(claimed_name: str) -> str:
    """Recover the original job filename from a claimed filename.

    Claimed files are named ``<original_name>___<hostname>``.  We split on
    the last occurrence of the triple-underscore separator.

    Falls back to the full claimed_name if the separator is not found
    (should not happen in normal operation).
    """
    idx = claimed_name.rfind(_CLAIM_SEP)
    if idx > 0:
        return claimed_name[:idx]
    # Fallback: return as-is (should not happen).
    log.warning("Could not recover original name from %r — using as-is", claimed_name)
    return claimed_name


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def _kill_process_tree(pid: int) -> None:
    """Send SIGKILL to *pid* and all its descendants (best-effort).

    Strategy (layered for reliability):
        1. Walk /proc to find all descendant PIDs (handles children that
           called setsid() and left the process group).
        2. Kill the entire process group via os.killpg() (catches most cases).
        3. Kill each individual descendant PID found in step 1.
        4. Kill the lead process directly as a final fallback.
    """
    # --- Step 1: Collect all descendant PIDs by walking /proc ---
    descendants: list[int] = []
    try:
        all_pids = [int(d) for d in os.listdir("/proc") if d.isdigit()]
        # Build parent->children map.
        children_map: dict[int, list[int]] = {}
        for p in all_pids:
            try:
                with open(f"/proc/{p}/stat", "r") as f:
                    stat_content = f.read()
                    # The comm field (field 2) is in parens and can contain
                    # spaces, parens, and arbitrary chars.  Find the last ')'
                    # to reliably locate the fields after it.
                    close_paren = stat_content.rfind(")")
                    if close_paren < 0:
                        continue
                    # Fields after ')': state ppid pgrp session ...
                    rest_fields = stat_content[close_paren + 2:].split()
                    ppid = int(rest_fields[1])  # index 0=state, 1=ppid
                    children_map.setdefault(ppid, []).append(p)
            except (FileNotFoundError, IndexError, ValueError, PermissionError):
                continue
        # BFS from pid to collect all descendants.
        queue = [pid]
        while queue:
            current = queue.pop(0)
            for child in children_map.get(current, []):
                descendants.append(child)
                queue.append(child)
    except (FileNotFoundError, OSError):
        pass  # /proc not available (non-Linux); fall through to killpg.

    # --- Step 2: Kill the process group ---
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass

    # --- Step 3: Kill each descendant individually (catches setsid'd children) ---
    for dpid in descendants:
        try:
            os.kill(dpid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    # --- Step 4: Kill the lead process directly ---
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def execute_job(
    job_path: Path,
    logs_dir: Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> int:
    """Run a claimed job script and return the exit code (or -1 on timeout).

    stdout and stderr are captured to ``logs/<claimed_name>.log``.
    The subprocess inherits the caller's full environment (including
    CUDA_VISIBLE_DEVICES and friends).

    On timeout the subprocess tree is killed aggressively to free GPUs.
    """
    log_file = logs_dir / f"{job_path.name}.log"

    log.info("Executing %s (timeout=%ds, log=%s)", job_path.name, timeout, log_file)

    # Recover the original filename (before hostname was appended) to determine
    # how to invoke the job.  The triple-underscore separator makes this robust
    # even when the original filename or hostname contains single underscores.
    original_name = _recover_original_name(job_path.name)
    original_suffix = Path(original_name).suffix.lower()

    if original_suffix in (".sh", ".bash", ""):
        cmd = ["bash", str(job_path)]
    elif original_suffix == ".py":
        cmd = [sys.executable, str(job_path)]
    elif original_suffix == ".json":
        # JSON job definitions: expect a wrapper; for now just cat and skip.
        log.warning("JSON job definitions not yet supported; skipping %s", job_path.name)
        return 1
    else:
        cmd = ["bash", str(job_path)]  # default: treat as shell script

    start = time.monotonic()
    with open(log_file, "w") as fh:
        # Write a header into the log for easy identification.
        fh.write(f"# Job:     {original_name}\n")
        fh.write(f"# Host:    {HOSTNAME}\n")
        fh.write(f"# Started: {datetime.now(timezone.utc).isoformat()}\n")
        fh.write(f"# Timeout: {timeout}s\n")
        fh.write(f"# Command: {' '.join(cmd)}\n")
        fh.write("#" + "-" * 72 + "\n")
        fh.flush()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),  # inherit CUDA_VISIBLE_DEVICES etc.
                start_new_session=True,  # own process group for clean kill
            )
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                log.warning(
                    "Job %s TIMED OUT after %.0fs — killing process tree (pid %d)",
                    job_path.name,
                    elapsed,
                    proc.pid,
                )
                _kill_process_tree(proc.pid)
                # Reap the zombie; use a try/except so a stuck zombie doesn't
                # crash the entire watcher loop.
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    log.error(
                        "Could not reap zombie for job %s (pid %d) after 30s",
                        job_path.name,
                        proc.pid,
                    )
                fh.write(f"\n# TIMEOUT after {elapsed:.0f}s — killed by watcher\n")
                return -1
        except Exception as exc:
            log.exception("Failed to execute %s: %s", job_path.name, exc)
            fh.write(f"\n# EXECUTION ERROR: {exc}\n")
            return 1

    elapsed = time.monotonic() - start
    log.info("Job %s finished in %.1fs with exit code %d", job_path.name, elapsed, returncode)
    return returncode


# ---------------------------------------------------------------------------
# Main watcher loop
# ---------------------------------------------------------------------------


def _unique_dest(target_dir: Path, name: str) -> Path:
    """Return a non-colliding destination path under *target_dir*.

    If a file with *name* already exists (e.g. a re-submitted job with the
    same name was previously completed), append a timestamp to avoid silently
    overwriting the earlier result.
    """
    dest = target_dir / name
    if not dest.exists():
        return dest
    stem = Path(name).stem
    suffix = Path(name).suffix
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = target_dir / f"{stem}_{ts}{suffix}"
    counter = 1
    while dest.exists():
        dest = target_dir / f"{stem}_{ts}_{counter}{suffix}"
        counter += 1
    return dest


def _safe_mtime(p: Path) -> float:
    """Return mtime of *p*, or inf if stat fails (so missing files sort last)."""
    try:
        return p.stat().st_mtime
    except OSError:
        return float("inf")


def watcher_loop(
    job_dir: str | Path,
    poll_interval: int = DEFAULT_POLL_SECONDS,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_concurrent: int = 1,
) -> None:
    """Poll pending/ and execute jobs forever.

    Currently runs jobs sequentially (max_concurrent=1).  The parameter is
    reserved for future multi-slot support.
    """
    dirs = init_job_dirs(job_dir)
    pending_dir = dirs["pending"]
    running_dir = dirs["running"]
    completed_dir = dirs["completed"]
    failed_dir = dirs["failed"]
    logs_dir = dirs["logs"]

    log.info(
        "Watcher started on %s — polling %s every %ds (timeout=%ds)",
        HOSTNAME,
        pending_dir,
        poll_interval,
        timeout,
    )

    while True:
        # List pending jobs sorted by modification time (oldest first = FIFO).
        # Use _safe_mtime to avoid crashing if a file vanishes between
        # iterdir() and stat() (race with another watcher node).
        try:
            candidates = sorted(pending_dir.iterdir(), key=_safe_mtime)
        except OSError as exc:
            log.warning("Error listing pending dir: %s", exc)
            candidates = []

        for candidate in candidates:
            # Skip dotfiles and non-files.
            if candidate.name.startswith(".") or not candidate.is_file():
                continue

            # --- Security: reject filenames with path separators ---
            try:
                _sanitize_filename(candidate.name)
            except ValueError:
                log.warning("Skipping invalid job filename: %r", candidate.name)
                continue

            # --- Atomic claim attempt (NFS-safe hardlink protocol) ---
            claimed = claim_job(candidate, running_dir)
            if claimed is None:
                continue  # another node grabbed it

            # --- Execute ---
            returncode = execute_job(claimed, logs_dir, timeout=timeout)

            # --- Cleanup: move to completed/ or failed/ ---
            # Wrap in try/except so a failed move doesn't crash the loop.
            target_dir = completed_dir if returncode == 0 else failed_dir
            label = "completed" if returncode == 0 else "failed"
            dest = _unique_dest(target_dir, claimed.name)
            try:
                shutil.move(str(claimed), str(dest))
                log.info("Job %s -> %s/ (exit=%d)", claimed.name, label, returncode)
            except OSError as exc:
                log.error(
                    "Failed to move job %s to %s/ (exit=%d): %s — "
                    "attempting cleanup of stale running/ file",
                    claimed.name,
                    label,
                    returncode,
                    exc,
                )
                # Best-effort: remove the stale file from running/ so it
                # doesn't accumulate.  If this also fails, log it but move on.
                try:
                    claimed.unlink(missing_ok=True)
                    log.info("Removed stale running/ file: %s", claimed.name)
                except OSError as cleanup_exc:
                    log.error(
                        "Could not clean up stale running/ file %s: %s",
                        claimed.name,
                        cleanup_exc,
                    )

        # Sleep before next poll.
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DFS job-queue watcher daemon (spool-directory pattern).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
    # Start inside a tmux session on each node:
    python -m uutils.job_scheduler_uu.scheduler --job-dir ~/dfs/job_queue --poll 15

    # Submit a job from any node:
    cp my_train.sh ~/dfs/job_queue/pending/
""",
    )
    parser.add_argument(
        "--job-dir",
        default=DEFAULT_JOB_DIR,
        help=f"Root of the spool directory tree (default: {DEFAULT_JOB_DIR})",
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help=f"Seconds between polls of pending/ (default: {DEFAULT_POLL_SECONDS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Max seconds per job before kill (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args()

    try:
        watcher_loop(
            job_dir=args.job_dir,
            poll_interval=args.poll,
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        log.info("Watcher stopped by user (Ctrl-C).")
        sys.exit(0)


if __name__ == "__main__":
    main()
