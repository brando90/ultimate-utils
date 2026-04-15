"""
Distributed task execution for SNAP cluster servers via shared filesystems (AFS/DFS/NFS).

The SNAP servers at Stanford (Jure Leskovec's group, https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-servers)
are individual named machines (rambo, madmax1-7, hyperion1-3, turing1-3, ampere1-9, etc.)
that you SSH into directly. They share:
    - /dfs/  -- NFS-mounted home dirs and scratch (accessible from all servers)
    - /afs/cs.stanford.edu/ -- AFS (Stanford CS department, Kerberos-authenticated)
    - /lfs/  -- Local fast storage per machine (NOT shared)

Some servers now have Slurm, but many don't, so you have to SSH into each one individually.
This module implements a filesystem-based task queue on the shared /dfs/ or /afs/ filesystem
that lets you submit tasks from anywhere and have them picked up by watcher daemons running
in tmux on each server. This is a "poor man's Slurm" using atomic file renames as locks.

Architecture:
    - Shared queue directory on /dfs/ or /afs/ (e.g., ~/afs_task_queue/)
    - Tasks are JSON files dropped into pending/
    - Each SNAP server runs a watcher daemon (in tmux) polling for new tasks
    - Watcher claims a task via atomic rename (prevents double-execution)
    - Watcher executes the task (e.g., runs Claude Code with --dangerously-skip-permissions)
    - Completed/failed tasks are moved to completed/ or failed/ with logs

Usage:
    # Submit a task from any machine with shared filesystem access:
    python -m uutils.snap_cluster.submit --prompt "Fix the bug in train.py" --queue_dir ~/afs_task_queue

    # Start a watcher daemon on a SNAP server (run inside tmux):
    python -m uutils.snap_cluster.watcher --queue_dir ~/afs_task_queue

    # Bootstrap watchers on all SNAP servers at once:
    bash py_src/uutils/snap_cluster/setup_watchers.sh

    # Check queue status:
    python -m uutils.snap_cluster.submit --status --queue_dir ~/afs_task_queue

Ref: https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-servers
"""
