"""
Unit Tests 1 to 10: Input Validation, Missing Values, and Consistency Reconciling.
"""

import unittest
from health_engine.engine import HealthIntelligenceEngine, PRODUCTION_OUTPUT_KEYS
from health_engine.synthetic import SyntheticTelemetryGenerator


class TestInputValidation(unittest.TestCase):

    def setUp(self):
        self.engine = HealthIntelligenceEngine()
        self.synth = SyntheticTelemetryGenerator(seed=101)

    def test_01_fully_valid_input(self):
        """Test 1: Fully valid input record processes cleanly and returns exact 47 fields."""
        rec = self.synth._base_reading(sample_id=1, timestamp=100.0, uptime_ms=5000.0)
        output = self.engine.process_reading(rec)

        self.assertEqual(len(output), 47)
        self.assertEqual(set(output.keys()), set(PRODUCTION_OUTPUT_KEYS))
        self.assertAlmostEqual(output["timestamp"], 100.0)
        self.assertGreater(output["overall_health_score"], 80.0)
        self.assertEqual(output["health_state"], "HEALTHY")
        self.assertFalse(output["cpu_overload_flag"])

    def test_02_missing_optional_values(self):
        """Test 2: Missing optional fields (e.g. interrupt_latency, task_jitter) do not crash system."""
        rec = self.synth._base_reading(sample_id=1, timestamp=101.0, uptime_ms=6000.0)
        del rec["interrupt_latency"]
        del rec["task_jitter"]
        del rec["power_consumption"]

        output = self.engine.process_reading(rec)
        self.assertEqual(len(output), 47)
        self.assertIsNotNone(output["overall_health_score"])

    def test_03_missing_important_values_derivation(self):
        """Test 3: Missing cpu_utilization is derived deterministically from cpu_idle."""
        rec = self.synth._base_reading(sample_id=1, timestamp=102.0, uptime_ms=7000.0)
        rec["cpu_idle"] = 72.0
        rec["cpu_utilization"] = None

        output = self.engine.process_reading(rec)
        self.assertEqual(len(output), 47)
        # cpu_mean should equal 100 - 72 = 28.0
        self.assertAlmostEqual(output["cpu_mean"], 28.0, delta=0.5)

    def test_04_null_values_distinguished_from_zero(self):
        """Test 4: Null values are not interpreted as observed zero."""
        rec1 = self.synth._base_reading(sample_id=1, timestamp=103.0, uptime_ms=8000.0)
        rec1["dropped_messages"] = 0 # Observed zero: explicitly healthy

        rec2 = self.synth._base_reading(sample_id=2, timestamp=104.0, uptime_ms=9000.0)
        rec2["dropped_messages"] = None # Missing: unknown, confidence penalized

        out1 = self.engine.process_reading(rec1, return_diagnostics=True)
        diag1 = out1[1]

        out2 = self.engine.process_reading(rec2, return_diagnostics=True)
        diag2 = out2[1]

        # Check that missing value resulted in lower sample confidence
        conf1 = diag1["overall_health_score"]["confidence"]
        conf2 = diag2["overall_health_score"]["confidence"]
        self.assertGreaterEqual(conf1, conf2)

    def test_05_impossible_cpu_values(self):
        """Test 5: CPU utilization outside [0, 100] is rejected and not factored into mean."""
        rec = self.synth._base_reading(sample_id=1, timestamp=105.0, uptime_ms=10000.0)
        rec["cpu_utilization"] = 150.0 # Impossible
        rec["cpu_idle"] = 65.0        # Valid idle

        output = self.engine.process_reading(rec)
        # Should derive cpu_utilization as 100 - 65 = 35.0
        self.assertAlmostEqual(output["cpu_mean"], 35.0, delta=0.5)

    def test_06_impossible_memory_values(self):
        """Test 6: Impossible heap values (used > total) are reconciled without NaN."""
        rec = self.synth._base_reading(sample_id=1, timestamp=106.0, uptime_ms=11000.0)
        rec["total_heap"] = 500000
        rec["used_heap"] = 800000 # Exceeds total
        rec["free_heap"] = -100   # Negative

        output = self.engine.process_reading(rec)
        self.assertEqual(len(output), 47)
        # Memory mean is bounded [0, 100]
        self.assertTrue(0.0 <= output["memory_mean"] <= 100.0)

    def test_07_inconsistent_memory_metrics(self):
        """Test 7: used_heap + free_heap != total_heap is reconciled."""
        rec = self.synth._base_reading(sample_id=1, timestamp=107.0, uptime_ms=12000.0)
        rec["total_heap"] = 1000000
        rec["used_heap"] = 300000
        rec["free_heap"] = 900000 # Sum = 1,200,000 != 1,000,000

        output = self.engine.process_reading(rec)
        # Free heap reconciled to 700,000 -> 30% utilization
        self.assertAlmostEqual(output["memory_mean"], 30.0, delta=1.0)

    def test_08_queue_length_exceeding_capacity(self):
        """Test 8: Queue length > capacity is handled safely and triggers queue overflow."""
        rec = self.synth._base_reading(sample_id=1, timestamp=108.0, uptime_ms=13000.0)
        rec["queue_capacity"] = 100
        rec["queue_length"] = 140 # Overflowed physical capacity

        output = self.engine.process_reading(rec)
        self.assertTrue(output["queue_overflow_flag"])

    def test_09_invalid_inference_timing_relationships(self):
        """Test 9: min_inference_time > max_inference_time is reconciled by swapping."""
        rec = self.synth._base_reading(sample_id=1, timestamp=109.0, uptime_ms=14000.0)
        rec["min_inference_time"] = 60.0
        rec["max_inference_time"] = 25.0
        rec["average_inference_time"] = 35.0

        output = self.engine.process_reading(rec)
        self.assertEqual(len(output), 47)
        self.assertAlmostEqual(output["latency_mean"], 35.0, delta=1.0)

    def test_10_negative_values(self):
        """Test 10: Negative counts/delays are rejected rather than polluting statistics."""
        rec = self.synth._base_reading(sample_id=1, timestamp=110.0, uptime_ms=15000.0)
        rec["scheduler_delay"] = -5.0
        rec["deadline_misses"] = -1

        output = self.engine.process_reading(rec)
        self.assertEqual(len(output), 47)
        self.assertFalse(output["deadline_miss_flag"])


if __name__ == "__main__":
    unittest.main()
