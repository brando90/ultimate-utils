"""Local regression checks for guard contention; all SSH/status calls are mocked."""

import subprocess
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
