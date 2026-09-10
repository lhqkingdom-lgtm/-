from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch

from memory_lab.cycle import watch_memory_cycles


class CycleTests(unittest.TestCase):
    def test_watch_has_bounded_test_mode_and_reports_each_cycle(self):
        observed = []

        def fake_cycle(*args, **kwargs):
            return {"status": "complete", "number": len(observed) + 1}

        with patch("memory_lab.cycle.source_activity_token", return_value="one"):
            result = watch_memory_cycles(
                Path("source.db"),
                Path("lab.db"),
                memory_root=Path("memory"),
                report_directory=Path("reports"),
                interval_seconds=1,
                max_cycles=1,
                on_cycle=observed.append,
                cycle_runner=fake_cycle,
            )
        self.assertEqual(result["cycle_count"], 1)
        self.assertEqual(len(observed), 1)


if __name__ == "__main__":
    unittest.main()
