"""Repro for issue #39 without sending email.

Builds a minimal PATH whose only ``clauded`` is the real SNAP wrapper
(/dfs/scratch0/brando9/bin/clauded, whose first line is '#\\!/bin/bash', not a
shebang), then calls the scheduler's lifecycle-email function with
``subprocess.Popen`` replaced by a recorder that only *checks* exec-readiness by
running ``os.execv`` in a forked child on ``--version``. No agent prompt ever
reaches an agent, so no email can be sent.
Usage: PYTHONPATH=<worktree>/py_src python repro_39.py
"""
import logging, os, subprocess, sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
# Minimal cron-like PATH: the broken DFS wrapper is the only `clauded`, as in the
# @reboot/cron watcher environment. (With the good /afs wrapper later on PATH,
# CPython silently skips the ENOEXEC entry and the bug hides.)
os.environ["PATH"] = "/dfs/scratch0/brando9/bin:/usr/local/bin:/usr/bin:/bin"
from uutils.job_scheduler_uu import scheduler  # noqa: E402

real_popen = subprocess.Popen
launched = []

def guarded_popen(cmd, *a, **kw):
    # Exec the same binary with --version only: proves exec-readiness, sends nothing.
    launched.append(cmd[0])
    return real_popen([cmd[0], "--version"], *a, **kw)

scheduler.subprocess.Popen = guarded_popen
print("which clauded ->", scheduler.shutil.which("clauded"))
print("agent resolved ->", scheduler._find_agent_binary())
for dry in ("1", ""):
    os.environ["UUTILS_WATCHER_NOTIFY_DRY_RUN"] = dry
    launched.clear()
    print(f"--- UUTILS_WATCHER_NOTIFY_DRY_RUN={dry!r}")
    scheduler._send_daemon_lifecycle_email("REPRO STARTED", "repro for issue #39")
    print("exec attempted on:", launched or "<nothing: dry-run or no agent>")
