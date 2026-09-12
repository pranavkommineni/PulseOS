"""
Unit Tests 11 to 16: Temporal Dynamics, Device Restarts, and Startup Behavior.
"""

import unittest
from health_engine.engine import HealthIntelligenceEngine
from health_engine.synthetic import SyntheticTelemetryGenerator


class TestTemporalDynamics(unittest.TestCase):

    def setUp(self):
        self.engine = HealthIntelligenceEngine()
        self.synth = SyntheticTelemetryGenerator(seed=202)

    def test_11_duplicate_timestamps(self):
        """Test 11: Consecutive samples with identical timestamps do not cause division by zero."""
        rec1 = self.synth._base_reading(sample_id=1, timestamp=200.0, uptime_ms=1000.0)
        rec2 = self.synth._base_reading(sample_id=2, timestamp=200.0, uptime_ms=1000.0)

        out1 = self.engine.process_reading(rec1)
        out2 = self.engine.process_reading(rec2)

        self.assertIsNotNone(out2["overall_health_score"])
        self.assertIn(out2["cpu_trend"], ["INSUFFICIENT_DATA", "STABLE"])

    def test_12_out_of_order_timestamps(self):
        """Test 12: Negative timestamp jump is identified as time discontinuity/reset."""
        rec1 = self.synth._base_reading(sample_id=1, timestamp=500.0, uptime_ms=10000.0)
        rec2 = self.synth._base_reading(sample_id=2, timestamp=450.0, uptime_ms=11000.0) # Jump backward

        self.engine.process_reading(rec1)
        out2 = self.engine.process_reading(rec2)

        self.assertTrue(self.engine.history.reset_detected)
        self.assertIn("NEGATIVE_TIME_JUMP", str(self.engine.history.reset_reason))

    def test_13_device_restart_detection(self):
        """Test 13: Uptime dropping from 60000ms to 500ms triggers clean buffer reset."""
        # Warm up with 10 samples
        for i in range(10):
            r = self.synth._base_reading(sample_id=i+1, timestamp=1000.0 + i, uptime_ms=60000.0 + i*1000)
            self.engine.process_reading(r)

        self.assertEqual(self.engine.history.history_length(), 10)

        # Reboot reading arrives
        reboot_rec = self.synth._base_reading(sample_id=1, timestamp=1020.0, uptime_ms=500.0)
        out_reboot = self.engine.process_reading(reboot_rec)

        self.assertTrue(self.engine.history.reset_detected)
        self.assertEqual(self.engine.history.history_length(), 1)
        self.assertEqual(out_reboot["cpu_trend"], "INSUFFICIENT_DATA")

    def test_14_counter_reset_detection(self):
        """Test 14: Incrementing watchdog_resets counter triggers reset handling."""
        r1 = self.synth._base_reading(sample_id=1, timestamp=100.0, uptime_ms=5000.0)
        r1["watchdog_resets"] = 0
        self.engine.process_reading(r1)

        r2 = self.synth._base_reading(sample_id=2, timestamp=101.0, uptime_ms=6000.0)
        r2["watchdog_resets"] = 1 # Watchdog tripped
        self.engine.process_reading(r2)

        self.assertTrue(self.engine.history.reset_detected)
        self.assertIn("WATCHDOG_RESET_INCREMENT", str(self.engine.history.reset_reason))

    def test_15_first_sample_behavior(self):
        """Test 15: Single initial reading outputs valid stats and INSUFFICIENT_DATA trends."""
        rec = self.synth._base_reading(sample_id=1, timestamp=50.0, uptime_ms=1000.0)
        rec["cpu_utilization"] = 42.0

        out = self.engine.process_reading(rec)
        self.assertEqual(out["cpu_mean"], 42.0)
        self.assertEqual(out["cpu_std"], 0.0)
        self.assertEqual(out["cpu_min"], 42.0)
        self.assertEqual(out["cpu_max"], 42.0)
        self.assertEqual(out["cpu_trend"], "INSUFFICIENT_DATA")
        self.assertIsNone(out["cpu_growth_rate"])

    def test_16_insufficient_history_points(self):
        """Test 16: With only 3 readings (below min_trend_points=5), trends remain INSUFFICIENT_DATA."""
        for i in range(3):
            r = self.synth._base_reading(sample_id=i+1, timestamp=10.0 + i, uptime_ms=1000.0 + i*1000)
            out = self.engine.process_reading(r)

        self.assertEqual(out["cpu_trend"], "INSUFFICIENT_DATA")
        self.assertEqual(out["latency_trend"], "INSUFFICIENT_DATA")
        self.assertIsNone(out["cpu_growth_rate"])


if __name__ == "__main__":
    unittest.main()
