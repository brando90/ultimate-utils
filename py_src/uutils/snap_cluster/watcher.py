"""
Watcher daemon that runs on each SNAP server (inside tmux).

Polls the shared AFS/DFS task queue for pending tasks, claims them atomically,
and executes them (e.g., running Claude Code with full permissions).

Usage:
    # Start in a tmux session on a SNAP server:
    python -m uutils.snap_cluster.watcher --queue_dir ~/afs_task_queue

    # With custom poll interval and server name:
    python -m uutils.snap_cluster.watcher \
        --queue_dir /afs/cs.stanford.edu/u/brando9/task_queue \
        --poll_interval 10 \
        --server_name snap5

    # Only accept tasks targeted at this server:
    python -m uutils.snap_cluster.watcher --queue_dir ~/afs_task_queue --only_targeted
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from uutils.snap_cluster.task_queue import (
    Task,
    init_queue,
    list_pending,
    claim_task,
    complete_task,
    get_server_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("snap_watcher")

# Graceful shutdown
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.info(f"Received signal {signum}, shutting down after current task...")
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# -- Executors -------------------------------------------------------------------

def execute_claude_code(task: Task, log_dir: Path) -> int:
    """
    Execute a task using Claude Code CLI.

    Runs: claude --print --dangerously-skip-permissions --model <model> "<prompt>"
    """
    stdout_log = log_dir / f"{task.task_id}.stdout.log"
    stderr_log = log_dir / f"{task.task_id}.stderr.log"

    cmd = ["claude"]

    # Permission mode
    if task.claude_permissions == "full":
        cmd.append("--dangerously-skip-permissions")

    # Model
    if task.claude_model:
        cmd.extend(["--model", task.claude_model])

    # Extra args
    if task.claude_extra_args:
        cmd.extend(task.claude_extra_args.split())

    # Use --print for non-interactive mode with the prompt
    cmd.extend(["--print", task.prompt])

    # Working directory
    cwd = os.path.expanduser(task.working_dir)
    if not os.path.isdir(cwd):
        log.warning(f"Working dir {cwd} doesn't exist, using home")
        cwd = os.path.expanduser("~")

    # Environment
    env = os.environ.copy()
    env.update(task.env)

    log.info(f"Running Claude Code: {' '.join(cmd[:6])}...")
    log.info(f"  cwd: {cwd}")
    log.info(f"  stdout -> {stdout_log}")
    log.info(f"  stderr -> {stderr_log}")

    with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=out,
            stderr=err,
            timeout=7200,  # 2 hour timeout per task
        )
    return proc.returncode


def execute_bash(task: Task, log_dir: Path) -> int:
    """Execute a task as a bash command."""
    stdout_log = log_dir / f"{task.task_id}.stdout.log"
    stderr_log = log_dir / f"{task.task_id}.stderr.log"

    cwd = os.path.expanduser(task.working_dir)
    if not os.path.isdir(cwd):
        cwd = os.path.expanduser("~")

    env = os.environ.copy()
    env.update(task.env)

    log.info(f"Running bash: {task.prompt[:80]}...")

    with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
        proc = subprocess.run(
            ["bash", "-c", task.prompt],
            cwd=cwd,
            env=env,
            stdout=out,
            stderr=err,
            timeout=7200,
        )
    return proc.returncode


def execute_python(task: Task, log_dir: Path) -> int:
    """Execute a task as a Python script."""
    stdout_log = log_dir / f"{task.task_id}.stdout.log"
    stderr_log = log_dir / f"{task.task_id}.stderr.log"

    cwd = os.path.expanduser(task.working_dir)
    if not os.path.isdir(cwd):
        cwd = os.path.expanduser("~")

    env = os.environ.copy()
    env.update(task.env)

    log.info(f"Running python: {task.prompt[:80]}...")

    with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
        proc = subprocess.run(
            ["python", "-c", task.prompt],
            cwd=cwd,
            env=env,
            stdout=out,
            stderr=err,
            timeout=7200,
        )
    return proc.returncode


EXECUTORS = {
    "claude_code": execute_claude_code,
    "bash": execute_bash,
    "python": execute_python,
}


# -- Main loop -------------------------------------------------------------------

def watcher_loop(
    queue_dir: str,
    server_name: str,
    poll_interval: float = 5.0,
    only_targeted: bool = False,
    max_concurrent: int = 1,
):
    """
    Main watcher loop. Polls for pending tasks and executes them.

    Args:
        queue_dir: Path to the shared task queue on AFS/DFS.
        server_name: Identifier for this server (e.g., "snap5").
        poll_interval: Seconds between polls when idle.
        only_targeted: If True, only pick up tasks explicitly targeted at this server.
        max_concurrent: Max tasks to run concurrently (currently sequential only).
    """
    qd = init_queue(queue_dir)
    log_dir = qd / "logs"

    log.info(f"Watcher started on {server_name}")
    log.info(f"Queue directory: {qd}")
    log.info(f"Poll interval: {poll_interval}s")
    log.info(f"Only targeted: {only_targeted}")
    log.info("Waiting for tasks...")

    while not _shutdown:
        try:
            # Get pending tasks eligible for this server
            target = server_name if only_targeted else None
            pending = list_pending(queue_dir, target_server=target)

            # Filter: skip tasks targeted at a DIFFERENT server
            eligible = []
            for t in pending:
                if t.target_server is None or t.target_server == server_name:
                    eligible.append(t)

            if not eligible:
                time.sleep(poll_interval)
                continue

            # Try to claim the highest-priority task
            for task in eligible:
                if claim_task(queue_dir, task, server_name):
                    log.info(f"Claimed task {task.task_id} (executor={task.executor})")
                    log.info(f"  prompt: {task.prompt[:120]}...")

                    # Execute
                    executor_fn = EXECUTORS.get(task.executor)
                    if executor_fn is None:
                        log.error(f"Unknown executor: {task.executor}")
                        complete_task(queue_dir, task, exit_code=1,
                                      error=f"Unknown executor: {task.executor}")
                        break

                    try:
                        exit_code = executor_fn(task, log_dir)
                        error_msg = None
                        if exit_code != 0:
                            # Read stderr for error details
                            stderr_file = log_dir / f"{task.task_id}.stderr.log"
                            if stderr_file.exists():
                                error_msg = stderr_file.read_text()[-2000:]  # last 2000 chars
                        log.info(f"Task {task.task_id} finished with exit_code={exit_code}")
                        complete_task(queue_dir, task, exit_code, error_msg)
                    except subprocess.TimeoutExpired:
                        log.error(f"Task {task.task_id} timed out")
                        complete_task(queue_dir, task, exit_code=-1, error="Task timed out (2h limit)")
                    except Exception as e:
                        log.error(f"Task {task.task_id} raised exception: {e}")
                        complete_task(queue_dir, task, exit_code=-1, error=str(e))

                    break  # Process one task at a time, then re-poll

        except Exception as e:
            log.error(f"Watcher error: {e}")
            time.sleep(poll_interval)

    log.info("Watcher shutting down.")


# -- CLI -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SNAP cluster task watcher daemon. Run inside tmux on each SNAP server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (auto-detects server name from hostname):
    python -m uutils.snap_cluster.watcher --queue_dir ~/afs_task_queue

    # Target a specific server with custom poll interval:
    python -m uutils.snap_cluster.watcher \\
        --queue_dir /afs/cs.stanford.edu/u/brando9/task_queue \\
        --poll_interval 10 \\
        --server_name snap5
        """,
    )
    parser.add_argument(
        "--queue_dir",
        type=str,
        default="~/afs_task_queue",
        help="Path to shared task queue directory on AFS/DFS (default: ~/afs_task_queue)",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default=None,
        help="Override server name (default: hostname)",
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=5.0,
        help="Seconds between polling for new tasks (default: 5)",
    )
    parser.add_argument(
        "--only_targeted",
        action="store_true",
        help="Only pick up tasks explicitly targeted at this server",
    )

    args = parser.parse_args()
    server_name = args.server_name or get_server_name()

    watcher_loop(
        queue_dir=args.queue_dir,
        server_name=server_name,
        poll_interval=args.poll_interval,
        only_targeted=args.only_targeted,
    )


if __name__ == "__main__":
    main()
