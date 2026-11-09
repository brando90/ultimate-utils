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
import time
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
HOST = "brando9@mercury2.stanford.edu"
SOURCE = "/lfs/mercury2/0/brando9/brier_ce_pilot_076e/lm_fixed"
GUARD = Path.home() / "agents-config/scripts/snap_reconnect.py"


def guard_status() -> dict:
    return json.loads(subprocess.check_output(["python3", str(GUARD), "--status"], timeout=10))


def reusable_connection() -> bool:
    result = subprocess.run(
        ["ssh", "-O", "check", HOST], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=10,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    if args.wait_seconds < 0:
        parser.error("--wait-seconds must be nonnegative")
    target = EXPERIMENT / "lm_fixed/state.json"
    if target.exists():
        print("Saved WikiText state already exists; no connection attempted.", flush=True)
        return 0
    deadline = time.monotonic() + args.wait_seconds
    last_next_attempt = None
    while True:
        status = guard_status()
        reused = reusable_connection()
        if reused or status["cooldown_remaining_seconds"] == 0:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0 or status["cooldown_remaining_seconds"] > remaining:
            print("RETRIEVAL_PENDING: shared cooldown exceeds this invocation's waiting budget. "
                  + json.dumps(status), flush=True)
            return 75
        if status["next_attempt"] != last_next_attempt:
            next_time = datetime.fromtimestamp(status["next_attempt"], ZoneInfo("America/Los_Angeles"))
            print("Waiting for existing connection or shared guard until "
                  + next_time.strftime("%m-%d-%Y %H:%M:%S %Z") + "; no network attempt made.", flush=True)
            last_next_attempt = status["next_attempt"]
        time.sleep(min(20, status["cooldown_remaining_seconds"], remaining))

    # The normal SSH configuration retains its own atomic guard; this status read
    # does not reserve a connection or bypass another client's concurrent attempt.
    status = guard_status()
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
    print("Reading existing WikiText state through configured SSH route.", flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=50)
    except subprocess.TimeoutExpired:
        print("RETRIEVAL_PENDING: bounded SSH read timed out; shared guard remains unchanged.", flush=True)
        return 75
    if result.returncode:
        print("RETRIEVAL_PENDING: SSH read failed with exit " + str(result.returncode)
              + "; no further network attempt in this invocation.", flush=True)
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
