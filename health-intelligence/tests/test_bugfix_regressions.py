"""
Regression tests for bugs found via fuzz/dry-run testing and fixed:

1. engine.process_reading({}) crashed with TypeError because a present-but-None
   "timestamp" key silently defeated dict.get(key, default), and the
   validator's is_valid=False signal was never checked.
2. faults.py crashed with TypeError when cpu_mean was None (cpu_utilization
   never observed) and context_switches was present, for the same
   get-with-default-vs-None reason.
3. fault_severity could report "HIGH" even when fault_count == 0 and
   overall_health_score was in the 90s, because the dominant-cause name check
   in _classify_severity bypassed the fault_count/overall_health guards.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from health_engine.engine import HealthIntelligenceEngine, PRODUCTION_OUTPUT_KEYS


class TestBugfixRegressions(unittest.TestCase):

    def test_empty_input_does_not_crash(self):
        """process_reading({}) must return a valid 47-key contract, not raise."""
        engine = HealthIntelligenceEngine()
        output = engine.process_reading({})
        self.assertEqual(set(output.keys()), set(PRODUCTION_OUTPUT_KEYS))
        self.assertEqual(output["health_state"], "UNKNOWN")
        self.assertEqual(output["fault_count"], 0)
        self.assertEqual(output["fault_type"], "NONE")

    def test_missing_temporal_anchor_is_inert_not_crashing(self):
        """A record with neither timestamp nor uptime_ms must not crash and
        must not be reported as a healthy/scored reading."""
        engine = HealthIntelligenceEngine()
        output = engine.process_reading({"cpu_utilization": 50.0})
        self.assertEqual(output["health_state"], "UNKNOWN")
        self.assertIsNone(output["overall_health_score"])

    def test_context_switches_with_unknown_cpu_mean_does_not_crash(self):
        """cpu_mean being None (cpu_utilization never observed) combined with
        context_switches present must not raise TypeError."""
        engine = HealthIntelligenceEngine()
        output = engine.process_reading({"timestamp": 1.0, "context_switches": 5})
        self.assertEqual(output["fault_type"], "NONE")

    def test_fault_severity_consistent_with_fault_count(self):
        """fault_severity must not be HIGH/CRITICAL when fault_count == 0 and
        overall_health_score is well above the medium/high thresholds."""
        engine = HealthIntelligenceEngine()
        output = None
        t = 0.0
        for i in range(8):
            cpu = 75 + i * 1.0  # slow ramp: stays under the overload flag threshold
            rec = {
                "timestamp": t,
                "cpu_utilization": cpu,
                "heap_utilization": 20.0,
                "stack_utilization": 20.0,
            }
            output = engine.process_reading(rec)
            t += 5.0

        self.assertEqual(output["fault_count"], 0)
        self.assertGreater(output["overall_health_score"], 85.0)
        self.assertNotIn(output["fault_severity"], ("HIGH", "CRITICAL"))

    def test_requirements_txt_is_pip_installable_syntax(self):
        """requirements.txt must contain only valid pip requirement lines
        (no bracketed non-extras annotations like '[optional]' or '[hjk]')."""
        req_path = PROJECT_ROOT / "requirements.txt"
        text = req_path.read_text()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            self.assertNotIn("[optional]", line)
            self.assertNotIn("[hjk]", line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
