"""
Submit tasks to the SNAP cluster task queue from any machine with AFS/DFS access.

Usage:
    # Submit a Claude Code task:
    python -m uutils.snap_cluster.submit --prompt "Fix the bug in train.py" --queue_dir ~/afs_task_queue

    # Submit to a specific server:
    python -m uutils.snap_cluster.submit --prompt "Run GPU test" --target snap5

    # Submit a bash command:
    python -m uutils.snap_cluster.submit --prompt "nvidia-smi" --executor bash

    # Submit with a specific working directory:
    python -m uutils.snap_cluster.submit --prompt "Run all tests" --working_dir ~/my-project

    # Check queue status:
    python -m uutils.snap_cluster.submit --status
"""
from __future__ import annotations

import argparse
import json
import sys

from uutils.snap_cluster.task_queue import (
    submit_task,
    queue_status,
    list_pending,
    list_running,
    list_completed,
    list_failed,
    init_queue,
    default_queue_dir,
)


def print_task_table(tasks, title: str):
    """Print a formatted table of tasks."""
    if not tasks:
        print(f"  {title}: (none)")
        return
    print(f"  {title} ({len(tasks)}):")
    for t in tasks:
        server = t.claimed_by or t.target_server or "any"
        prompt_preview = t.prompt[:60].replace("\n", " ")
        if t.exit_code is not None:
            status = f"exit={t.exit_code}"
        elif t.claimed_by:
            status = f"running on {t.claimed_by}"
        else:
            status = "pending"
        print(f"    [{t.task_id}] ({t.executor}) {prompt_preview}... -> {status}")


def show_status(queue_dir: str):
    """Show full queue status."""
    init_queue(queue_dir)
    status = queue_status(queue_dir)
    print(f"\nQueue: {queue_dir}")
    print(f"  Pending:   {status['pending']}")
    print(f"  Running:   {status['running']}")
    print(f"  Completed: {status['completed']}")
    print(f"  Failed:    {status['failed']}")
    print()

    print_task_table(list_pending(queue_dir), "Pending tasks")
    print_task_table(list_running(queue_dir), "Running tasks")
    print_task_table(list_completed(queue_dir, limit=5), "Recent completed")
    print_task_table(list_failed(queue_dir, limit=5), "Recent failed")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Submit tasks to the SNAP cluster task queue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Submit a Claude Code task:
    python -m uutils.snap_cluster.submit --prompt "Fix bug in train.py"

    # Submit to specific server:
    python -m uutils.snap_cluster.submit --prompt "Run GPU test" --target snap5

    # Submit a bash command:
    python -m uutils.snap_cluster.submit --prompt "nvidia-smi" --executor bash

    # Check status:
    python -m uutils.snap_cluster.submit --status
        """,
    )
    parser.add_argument("--prompt", type=str, help="Task prompt or command to execute")
    _default_qd = default_queue_dir()
    parser.add_argument(
        "--queue_dir",
        type=str,
        default=_default_qd,
        help=f"Path to shared task queue on DFS/AFS (default: {_default_qd})",
    )
    parser.add_argument(
        "--executor",
        type=str,
        default="claude_code",
        choices=["claude_code", "bash", "python"],
        help="How to execute the task (default: claude_code)",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        default="~",
        help="Working directory for task execution (default: ~)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target a specific server (e.g., snap5). Default: any available server",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Task priority (higher = picked up first, default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="opus",
        help="Claude model to use (default: opus)",
    )
    parser.add_argument(
        "--claude_extra_args",
        type=str,
        default="",
        help="Extra CLI arguments for Claude Code",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show queue status instead of submitting",
    )

    args = parser.parse_args()

    if args.status:
        show_status(args.queue_dir)
        return

    if not args.prompt:
        parser.error("--prompt is required when not using --status")

    task = submit_task(
        queue_dir=args.queue_dir,
        prompt=args.prompt,
        executor=args.executor,
        working_dir=args.working_dir,
        target_server=args.target,
        priority=args.priority,
        claude_model=args.model,
        claude_extra_args=args.claude_extra_args,
    )
    print(f"Task submitted: {task.task_id}")
    print(f"  Executor:    {task.executor}")
    print(f"  Target:      {task.target_server or 'any'}")
    print(f"  Working dir: {task.working_dir}")
    print(f"  Priority:    {task.priority}")
    print(f"  Prompt:      {task.prompt[:100]}...")
    print(f"\nCheck status: python -m uutils.snap_cluster.submit --status --queue_dir {args.queue_dir}")


if __name__ == "__main__":
    main()
