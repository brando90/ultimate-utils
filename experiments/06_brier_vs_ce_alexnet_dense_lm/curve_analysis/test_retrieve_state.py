"""Local regression checks for guard contention; all SSH/status calls are mocked."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import diagnostic_snap_proxy as diagnostic_proxy
import retrieve_wikitext_state as retrieval


class GuardContentionTests(unittest.TestCase):
    def test_busy_status_is_transient(self):
        result = subprocess.CompletedProcess(
            ["guard", "--status"], 255, "",
            "snap-reconnect: another SNAP connection is opening; no network attempt made\n",
        )
        with patch.object(retrieval.subprocess, "run", return_value=result):
            self.assertIsNone(retrieval.guard_status())

    def test_other_status_errors_remain_errors(self):
        result = subprocess.CompletedProcess(["guard", "--status"], 255, "", "invalid reconnect state")
        with patch.object(retrieval.subprocess, "run", return_value=result):
            with self.assertRaises(subprocess.CalledProcessError):
                retrieval.guard_status()

    def test_contention_then_waits_until_allowed_without_extra_status_call(self):
        now = [100.0]
        sleeps = []
        status = {"next_attempt": 102.0, "cooldown_remaining_seconds": 1}

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        with patch.object(retrieval, "guard_status", side_effect=[None, status]) as check, \
             patch.object(retrieval, "reusable_connection", return_value=False), \
             patch.object(retrieval.time, "time", side_effect=lambda: now[0]), \
             patch.object(retrieval.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(retrieval.time, "sleep", side_effect=sleep):
            self.assertEqual(retrieval.wait_for_admission(10), (status, False))
        self.assertEqual(now[0], 102.0)
        self.assertEqual(sleeps, [1, 1])
        self.assertEqual(check.call_count, 2)

    def test_busy_status_respects_zero_wait_budget(self):
        with patch.object(retrieval, "guard_status", return_value=None), \
             patch.object(retrieval, "reusable_connection") as reuse, \
             patch.object(retrieval.time, "sleep") as sleep:
            self.assertIsNone(retrieval.wait_for_admission(0))
        reuse.assert_not_called()
        sleep.assert_not_called()

    def test_existing_connection_can_be_reused_during_cooldown(self):
        status = {"next_attempt": 10**12, "cooldown_remaining_seconds": 3600}
        with patch.object(retrieval, "guard_status", return_value=status), \
             patch.object(retrieval, "reusable_connection", return_value=True), \
             patch.object(retrieval.time, "sleep") as sleep:
            self.assertEqual(retrieval.wait_for_admission(0), (status, True))
        sleep.assert_not_called()


class DiagnosticProxyTests(unittest.TestCase):
    def test_expected_gateway_command_changes_only_verbosity(self):
        original = diagnostic_proxy.EXPECTED_CHILD.copy()
        result = diagnostic_proxy.diagnostic_args(original)
        self.assertEqual(result[1], "-vv")
        self.assertEqual(result[:1] + result[2:], original)
        self.assertEqual(original, diagnostic_proxy.EXPECTED_CHILD)

    def test_unrecognized_child_is_unchanged(self):
        command = ["ssh", "some-other-host"]
        self.assertIs(diagnostic_proxy.diagnostic_args(command), command)

    def test_other_proxy_targets_are_rejected_before_execution(self):
        with patch.object(diagnostic_proxy.sys, "argv", ["proxy", "mercury1.stanford.edu", "22"]), \
             patch.object(diagnostic_proxy.runpy, "run_path") as run:
            with self.assertRaises(SystemExit):
                diagnostic_proxy.main()
        run.assert_not_called()

    def test_timeout_preserves_complete_partial_diagnostics(self):
        stderr = b"partial gateway diagnostic\n" * 300
        stdout = b"partial protocol output"
        status = {"next_attempt": 0, "cooldown_remaining_seconds": 0}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "private-diagnostics"
            with patch.object(retrieval, "EXPERIMENT", root), \
                 patch.object(retrieval.sys, "argv", ["retrieve", "--diagnostics-dir", str(destination)]), \
                 patch.object(retrieval, "wait_for_admission", return_value=(status, False)), \
                 patch.object(retrieval.subprocess, "run", side_effect=subprocess.TimeoutExpired(
                     ["ssh"], 50, output=stdout, stderr=stderr)) as run:
                self.assertEqual(retrieval.main(), 75)
            self.assertEqual((destination / "stderr.txt").read_bytes(), stderr)
            self.assertEqual((destination / "stdout.bin").read_bytes(), stdout)
            self.assertTrue(json.loads((destination / "receipt.json").read_text())["timed_out"])
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
