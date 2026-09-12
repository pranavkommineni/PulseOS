"""
Unit Tests 17 to 27: Operational Scenarios, Degradation, Faults, and Recovery.
"""

import unittest
from health_engine.engine import HealthIntelligenceEngine
from health_engine.synthetic import SyntheticTelemetryGenerator


class TestOperationalScenarios(unittest.TestCase):

    def setUp(self):
        self.engine = HealthIntelligenceEngine()
        self.synth = SyntheticTelemetryGenerator(seed=303)

    def test_17_stable_healthy_system(self):
        """Test 17: Normal healthy operation maintains high health score and zero faults."""
        records = self.synth.generate_scenario("HEALTHY", num_samples=15)
        for r in records:
            out = self.engine.process_reading(r)

        self.assertGreaterEqual(out["overall_health_score"], 85.0)
        self.assertEqual(out["health_state"], "HEALTHY")
        self.assertEqual(out["fault_type"], "NONE")
        self.assertEqual(out["fault_severity"], "NONE")
        self.assertEqual(out["fault_count"], 0)
        self.assertLess(out["degradation_index"], 0.20)

    def test_18_cpu_degradation(self):
        """Test 18: Sustained high CPU load triggers CPU overload flag and root cause diagnosis."""
        records = self.synth.generate_scenario("CPU_OVERLOAD", num_samples=20)
        for r in records:
            out = self.engine.process_reading(r)

        self.assertTrue(out["cpu_overload_flag"])
        self.assertEqual(out["cpu_trend"], "DEGRADING")
        self.assertEqual(out["fault_type"], "CPU_OVERLOAD")
        self.assertIn(out["fault_severity"], ["HIGH", "CRITICAL"])
        self.assertLess(out["cpu_health_score"], 50.0)

    def test_19_memory_leak_degradation(self):
        """Test 19: Monotonic heap growth triggers memory leak flag and positive growth rate."""
        records = self.synth.generate_scenario("MEMORY_LEAK", num_samples=20)
        for r in records:
            out = self.engine.process_reading(r)

        self.assertTrue(out["memory_leak_flag"])
        self.assertEqual(out["heap_trend"], "DEGRADING")
        self.assertIsNotNone(out["heap_growth_rate"])
        self.assertGreater(out["heap_growth_rate"], 0.0)
        self.assertIn(out["fault_type"], ["MEMORY_LEAK", "MEMORY_PRESSURE"])

    def test_20_heap_exhaustion(self):
        """Test 20: Critical heap utilization (>92%) triggers heap exhaustion and critical severity."""
        records = self.synth.generate_scenario("HEAP_EXHAUSTION", num_samples=15)
        for r in records:
            out = self.engine.process_reading(r)

        self.assertTrue(out["heap_exhaustion_flag"])
        self.assertEqual(out["fault_severity"], "CRITICAL")
        self.assertGreater(out["degradation_index"], 0.50)

    def test_21_stack_risk(self):
        """Test 21: High stack utilization (>88%) and low watermark trigger stack risk flag."""
        records = self.synth.generate_scenario("STACK_RISK", num_samples=15)
        for r in records:
            out = self.engine.process_reading(r)

        self.assertTrue(out["stack_risk_flag"])
        self.assertIn(out["fault_type"], ["STACK_RISK", "MULTIPLE_FAULTS"])
        self.assertLess(out["stack_health_score"], 50.0)

    def test_22_increasing_latency(self):
        """Test 22: AI inference latency inflation triggers ai_latency_flag and degrades AI score."""
        records = self.synth.generate_scenario("AI_LATENCY", num_samples=20)
        for r in records:
            out = self.engine.process_reading(r)

        self.assertTrue(out["ai_latency_flag"])
        self.assertEqual(out["latency_trend"], "DEGRADING")
        self.assertLess(out["ai_health_score"], 60.0)
        self.assertIn(out["fault_type"], ["AI_LATENCY", "AI_OVERLOAD", "CPU_OVERLOAD"])

    def test_23_deadline_misses(self):
        """Test 23: Task execution exceeding period triggers deadline_miss_flag and miss rate."""
        records = self.synth.generate_scenario("DEADLINE_DEGRADATION", num_samples=20)
        for r in records:
            out = self.engine.process_reading(r)

        self.assertTrue(out["deadline_miss_flag"])
        self.assertGreater(out["deadline_miss_rate"], 0.0)
        self.assertLess(out["timing_health_score"], 60.0)

    def test_24_queue_congestion(self):
        """Test 24: Queue filling to capacity and dropped messages trigger queue_overflow_flag."""
        records = self.synth.generate_scenario("QUEUE_CONGESTION", num_samples=20)
        for r in records:
            out = self.engine.process_reading(r)

        self.assertTrue(out["queue_overflow_flag"])
        self.assertEqual(out["queue_trend"], "DEGRADING")
        self.assertLess(out["resource_health_score"], 70.0)

    def test_25_task_starvation(self):
        """Test 25: Ready task blocked with excessive scheduler delay triggers task_starvation_flag."""
        records = self.synth.generate_scenario("TASK_STARVATION", num_samples=15)
        for r in records:
            out = self.engine.process_reading(r)

        self.assertTrue(out["task_starvation_flag"])
        self.assertIn(out["fault_type"], ["TASK_STARVATION", "SCHEDULING_OVERLOAD", "CPU_OVERLOAD"])

    def test_26_multiple_simultaneous_faults(self):
        """Test 26: Concurrent severe faults diagnosed as MULTIPLE_FAULTS with high fault count."""
        records = self.synth.generate_scenario("MULTIPLE_FAULTS", num_samples=20)
        for r in records:
            out = self.engine.process_reading(r)

        self.assertGreaterEqual(out["fault_count"], 2)
        self.assertEqual(out["fault_type"], "MULTIPLE_FAULTS")
        self.assertIn(out["fault_severity"], ["HIGH", "CRITICAL"])

    def test_27_recovery(self):
        """Test 27: System transitions from degraded state back towards healthy with positive health change."""
        records = self.synth.generate_scenario("RECOVERY", num_samples=24)
        outputs = []
        for r in records:
            outputs.append(self.engine.process_reading(r))

        mid_out = outputs[10] # During degradation
        end_out = outputs[-1] # After recovery

        self.assertLess(mid_out["overall_health_score"], end_out["overall_health_score"])
        self.assertGreater(mid_out["degradation_index"], end_out["degradation_index"])
        # During recovery phase, health change rate is positive (health improving)
        recov_sample = outputs[-3]
        if recov_sample["health_change_rate"] is not None:
            self.assertGreaterEqual(recov_sample["health_change_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
