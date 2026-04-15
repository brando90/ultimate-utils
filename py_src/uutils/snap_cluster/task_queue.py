"""
Filesystem-based task queue for distributing work across SNAP servers via AFS/DFS.

Uses atomic file operations (os.rename on the same filesystem) for locking,
so no external dependencies are needed -- just a shared filesystem.

Directory layout:
    queue_dir/
        pending/        # New tasks waiting to be picked up
        running/        # Tasks currently being executed (claimed by a server)
        completed/      # Successfully finished tasks (with logs)
        failed/         # Failed tasks (with error logs)
        logs/           # Stdout/stderr logs per task
"""
from __future__ import annotations

import json
import os
import time
import uuid
import platform
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List


# -- Task schema ----------------------------------------------------------------

@dataclass
class Task:
    """A unit of work to be dispatched to a SNAP server."""
    task_id: str
    prompt: str                          # The prompt / command to run
    executor: str = "claude_code"        # "claude_code", "bash", "python"
    working_dir: str = "~"               # cwd for execution
    env: dict = field(default_factory=dict)  # extra env vars
    target_server: Optional[str] = None  # None = any server, or e.g. "snap5"
    priority: int = 0                    # higher = picked up first
    created_at: float = field(default_factory=time.time)
    claimed_by: Optional[str] = None
    claimed_at: Optional[float] = None
    completed_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    # Claude Code specific options
    claude_model: str = "opus"           # model to use
    claude_permissions: str = "full"     # "full" = --dangerously-skip-permissions
    claude_extra_args: str = ""          # any extra CLI flags

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "Task":
        return cls(**json.loads(data))

    @classmethod
    def from_file(cls, path: Path) -> "Task":
        return cls.from_json(path.read_text())


# -- Queue operations -----------------------------------------------------------

def init_queue(queue_dir: str | Path) -> Path:
    """Create the queue directory structure. Idempotent."""
    qd = Path(queue_dir).expanduser().resolve()
    for sub in ("pending", "running", "completed", "failed", "logs"):
        (qd / sub).mkdir(parents=True, exist_ok=True)
    return qd


def submit_task(
    queue_dir: str | Path,
    prompt: str,
    executor: str = "claude_code",
    working_dir: str = "~",
    target_server: Optional[str] = None,
    priority: int = 0,
    claude_model: str = "opus",
    claude_permissions: str = "full",
    claude_extra_args: str = "",
    env: Optional[dict] = None,
) -> Task:
    """Submit a new task to the queue. Returns the Task object."""
    qd = init_queue(queue_dir)
    task = Task(
        task_id=f"{int(time.time())}_{uuid.uuid4().hex[:8]}",
        prompt=prompt,
        executor=executor,
        working_dir=working_dir,
        target_server=target_server,
        priority=priority,
        claude_model=claude_model,
        claude_permissions=claude_permissions,
        claude_extra_args=claude_extra_args,
        env=env or {},
    )
    task_file = qd / "pending" / f"{task.task_id}.json"
    # Write to a temp file first, then atomic rename, to avoid partial reads
    tmp_file = qd / "pending" / f".tmp_{task.task_id}.json"
    tmp_file.write_text(task.to_json())
    os.rename(str(tmp_file), str(task_file))
    return task


def list_pending(queue_dir: str | Path, target_server: Optional[str] = None) -> List[Task]:
    """List pending tasks, sorted by priority (desc) then creation time (asc)."""
    qd = Path(queue_dir).expanduser().resolve()
    pending_dir = qd / "pending"
    if not pending_dir.exists():
        return []
    tasks = []
    for f in pending_dir.glob("*.json"):
        try:
            t = Task.from_file(f)
            # Filter by target server if specified
            if target_server and t.target_server and t.target_server != target_server:
                continue
            # Also skip if task targets a specific server and we're not it
            if t.target_server and target_server and t.target_server != target_server:
                continue
            tasks.append(t)
        except Exception:
            continue  # skip malformed files
    tasks.sort(key=lambda t: (-t.priority, t.created_at))
    return tasks


def claim_task(queue_dir: str | Path, task: Task, server_name: str) -> bool:
    """
    Atomically claim a task by renaming it from pending/ to running/.
    Returns True if successful, False if another server grabbed it first.
    """
    qd = Path(queue_dir).expanduser().resolve()
    src = qd / "pending" / f"{task.task_id}.json"
    dst = qd / "running" / f"{task.task_id}.json"
    try:
        # Update task metadata before moving
        task.claimed_by = server_name
        task.claimed_at = time.time()
        # Atomic rename -- if src doesn't exist, another server got it
        os.rename(str(src), str(dst))
        # Now update the file contents with claim info
        dst.write_text(task.to_json())
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def complete_task(queue_dir: str | Path, task: Task, exit_code: int, error: Optional[str] = None):
    """Move a task from running/ to completed/ or failed/."""
    qd = Path(queue_dir).expanduser().resolve()
    src = qd / "running" / f"{task.task_id}.json"
    task.completed_at = time.time()
    task.exit_code = exit_code
    task.error = error
    dest_dir = "completed" if exit_code == 0 else "failed"
    dst = qd / dest_dir / f"{task.task_id}.json"
    try:
        src.write_text(task.to_json())
        os.rename(str(src), str(dst))
    except FileNotFoundError:
        # Task file was already moved/cleaned -- write directly
        dst.write_text(task.to_json())


def get_server_name() -> str:
    """Get a unique identifier for this server (hostname)."""
    return platform.node()


def list_running(queue_dir: str | Path) -> List[Task]:
    """List currently running tasks."""
    qd = Path(queue_dir).expanduser().resolve()
    running_dir = qd / "running"
    if not running_dir.exists():
        return []
    tasks = []
    for f in running_dir.glob("*.json"):
        try:
            tasks.append(Task.from_file(f))
        except Exception:
            continue
    return tasks


def list_completed(queue_dir: str | Path, limit: int = 20) -> List[Task]:
    """List recently completed tasks."""
    qd = Path(queue_dir).expanduser().resolve()
    completed_dir = qd / "completed"
    if not completed_dir.exists():
        return []
    tasks = []
    for f in sorted(completed_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            tasks.append(Task.from_file(f))
        except Exception:
            continue
    return tasks


def list_failed(queue_dir: str | Path, limit: int = 20) -> List[Task]:
    """List recently failed tasks."""
    qd = Path(queue_dir).expanduser().resolve()
    failed_dir = qd / "failed"
    if not failed_dir.exists():
        return []
    tasks = []
    for f in sorted(failed_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            tasks.append(Task.from_file(f))
        except Exception:
            continue
    return tasks


def queue_status(queue_dir: str | Path) -> dict:
    """Get a summary of the queue state."""
    qd = Path(queue_dir).expanduser().resolve()
    status = {}
    for sub in ("pending", "running", "completed", "failed"):
        d = qd / sub
        if d.exists():
            status[sub] = len(list(d.glob("*.json")))
        else:
            status[sub] = 0
    return status
