"""Retrieve existing WikiText state through the configured SSH guard; no model runs.

Default: reuse an existing connection or make one allowed attempt, otherwise exit 75.
--wait-seconds permits bounded waiting for the existing shared guard. It never resets
the guard, changes SSH configuration, or writes anything on the cluster.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
HOST = "brando9@mercury2.stanford.edu"
SOURCE = "/lfs/mercury2/0/brando9/brier_ce_pilot_076e/lm_fixed"
GUARD = Path.home() / "agents-config/scripts/snap_reconnect.py"


def guard_status() -> dict | None:
    result = subprocess.run(
        ["python3", str(GUARD), "--status"], capture_output=True, text=True, timeout=10,
    )
    if result.returncode:
        if result.returncode == 255 and "another SNAP connection is opening" in result.stderr:
            return None  # Transient local lock contention, not a transport failure.
        result.check_returncode()
    return json.loads(result.stdout)


def reusable_connection() -> bool:
    result = subprocess.run(
        ["ssh", "-O", "check", HOST], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=10,
    )
    return result.returncode == 0


def wait_for_admission(wait_seconds: int) -> tuple[dict, bool] | None:
    deadline = time.monotonic() + wait_seconds
    last_next_attempt = None
    status = None
    while True:
        # The status command was already run before this sleep. The configured
        # SSH proxy still makes the authoritative atomic admission decision.
        # Avoid repeated status/socket subprocesses at the exact opening time.
        if status is not None and time.time() >= status["next_attempt"]:
            return status, False
        status = guard_status()
        if status is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("RETRIEVAL_PENDING: local guard lock is busy; no network attempt made.", flush=True)
                return None
            time.sleep(min(1, remaining))
            continue
        reused = reusable_connection()
        delay = max(0.0, status["next_attempt"] - time.time())
        if reused or delay == 0:
            return status, reused
        remaining = deadline - time.monotonic()
        if remaining <= 0 or delay > remaining:
            print("RETRIEVAL_PENDING: shared cooldown exceeds this invocation's waiting budget. "
                  + json.dumps(status), flush=True)
            return None
        if status["next_attempt"] != last_next_attempt:
            next_time = datetime.fromtimestamp(status["next_attempt"], ZoneInfo("America/Los_Angeles"))
            print("Waiting for existing connection or shared guard until "
                  + next_time.strftime("%m-%d-%Y %H:%M:%S %Z") + "; no network attempt made.", flush=True)
            last_next_attempt = status["next_attempt"]
        time.sleep(min(20, delay, remaining))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--diagnostics-dir", type=Path,
                        help="New private local directory for full SSH diagnostics; adds verbosity only.")
    args = parser.parse_args()
    if args.wait_seconds < 0:
        parser.error("--wait-seconds must be nonnegative")
    target = EXPERIMENT / "lm_fixed/state.json"
    if target.exists():
        print("Saved WikiText state already exists; no connection attempted.", flush=True)
        return 0
    if args.diagnostics_dir is not None:
        args.diagnostics_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    admission = wait_for_admission(args.wait_seconds)
    if admission is None:
        return 75
    status, reused = admission

    # Status inspection does not reserve a connection or bypass another client.
    remote = f"""import base64,hashlib,json,pathlib
p=pathlib.Path({SOURCE!r})
s=(p/'state.json').read_bytes()
o={{'state_b64':base64.b64encode(s).decode(),'state_sha256':hashlib.sha256(s).hexdigest(),
   'files':[{{'name':f.name,'bytes':f.stat().st_size,'is_file':f.is_file()}} for f in sorted(p.iterdir())],
   'results_sha256':hashlib.sha256((p/'results.md').read_bytes()).hexdigest()}}
print(json.dumps(o))
"""
    cmd = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST,
           "python3 -c " + shlex.quote(remote)]
    if args.diagnostics_dir is not None:
        proxy = " ".join(shlex.quote(str(part)) for part in
                         (sys.executable, HERE / "diagnostic_snap_proxy.py")) + " %h %p"
        cmd[1:1] = ["-vv", "-o", "ProxyCommand=" + proxy]
    started_at = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()

    def save_diagnostics(stdout, stderr, returncode, timed_out=False):
        if args.diagnostics_dir is None:
            return
        for name, data in (("stdout.bin", stdout or b""), ("stderr.txt", stderr or b"")):
            with (args.diagnostics_dir / name).open("xb") as output:
                output.write(data)
        diagnostic = {
            "started_at": started_at,
            "ended_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
            "returncode": returncode, "timed_out": timed_out,
            "guard_status_before_attempt": status,
            "canonical_guard_sha256": hashlib.sha256(GUARD.read_bytes()).hexdigest(),
            "diagnostic_proxy_sha256": hashlib.sha256((HERE / "diagnostic_snap_proxy.py").read_bytes()).hexdigest(),
            "stdout_sha256": hashlib.sha256(stdout or b"").hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr or b"").hexdigest(),
            "scope": "Verbosity only; canonical shared guard, gateway route, authentication and host-key checks preserved.",
        }
        with (args.diagnostics_dir / "receipt.json").open("x") as output:
            output.write(json.dumps(diagnostic, indent=2) + "\n")

    print("Reading existing WikiText state through configured SSH route.", flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=50)
    except subprocess.TimeoutExpired as exc:
        save_diagnostics(exc.stdout, exc.stderr, None, timed_out=True)
        print("RETRIEVAL_PENDING: bounded SSH read timed out; shared guard and backoff remain in force.", flush=True)
        return 75
    save_diagnostics(result.stdout, result.stderr, result.returncode)
    if result.returncode:
        print("RETRIEVAL_PENDING: SSH read failed with exit " + str(result.returncode)
              + "; no further network attempt in this invocation.", flush=True)
        if args.diagnostics_dir is not None:
            print("Full connection diagnostics saved privately in " + str(args.diagnostics_dir), flush=True)
        else:
            print(result.stderr.decode(errors="replace")[-2000:], flush=True)
        return 75
    receipt = json.loads(result.stdout)
    raw = base64.b64decode(receipt.pop("state_b64"), validate=True)
    assert hashlib.sha256(raw).hexdigest() == receipt["state_sha256"]
    state = json.loads(raw)
    assert state["status"] == "DONE"
    assert state["protocol"]["final_steps"] == 57859
    assert state["protocol"]["corpus"] == "Salesforce/wikitext wikitext-103-raw-v1"
    assert len([r for r in state["records"] if r["phase"] == "final" and r["split"] == "validation"]) == 6
    local_results_sha256 = hashlib.sha256((EXPERIMENT / "lm_fixed/results.md").read_bytes()).hexdigest()
    assert receipt["results_sha256"] == local_results_sha256, "Remote results differ from frozen local results."
    receipt.update({
        "retrieved_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "source_host": HOST, "source_path": SOURCE + "/state.json",
        "operation": "Read-only retrieval of saved state and directory metadata; no remote writes, training, or evaluation.",
        "guard_before_attempt": status, "connection_reuse_check_succeeded": reused,
        "local_results_sha256": local_results_sha256,
    })
    with target.open("xb") as output:
        output.write(raw)
    (HERE / "retrieval_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("RETRIEVED: " + str(len(raw)) + " bytes; SHA-256 " + receipt["state_sha256"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
