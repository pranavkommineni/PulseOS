"""
Verification of JSON Output Contract, Field Invariance, and Diagnostic Isolation.
"""

import json
import math
import unittest
from health_engine.engine import HealthIntelligenceEngine, PRODUCTION_OUTPUT_KEYS
from health_engine.synthetic import SyntheticTelemetryGenerator


class TestContractCompliance(unittest.TestCase):

    def setUp(self):
        self.engine = HealthIntelligenceEngine()
        self.synth = SyntheticTelemetryGenerator(seed=404)

    def test_strict_47_field_contract(self):
        """Verify that every output dictionary contains exactly the 47 specified keys."""
        rec = self.synth._base_reading(sample_id=1, timestamp=100.0, uptime_ms=5000.0)
        output = self.engine.process_reading(rec)

        self.assertEqual(len(output), 47)
        self.assertEqual(len(PRODUCTION_OUTPUT_KEYS), 47)
        for k in PRODUCTION_OUTPUT_KEYS:
            self.assertIn(k, output, f"Missing required output key: {k}")

    def test_json_validity_and_no_nan_or_inf(self):
        """Verify output can be serialized to JSON and never contains NaN, Inf, or -Inf."""
        records = self.synth.generate_scenario("CPU_OVERLOAD", num_samples=10)
        for r in records:
            out = self.engine.process_reading(r)
            # JSON serialization check
            json_str = json.dumps(out)
            self.assertIsInstance(json_str, str)

            # Check individual values
            for k, v in out.items():
                if isinstance(v, float):
                    self.assertFalse(math.isnan(v), f"Key {k} contains NaN")
                    self.assertFalse(math.isinf(v), f"Key {k} contains Infinity")

    def test_diagnostic_isolation(self):
        """Verify that requesting diagnostics does not alter the production output contract."""
        rec = self.synth._base_reading(sample_id=1, timestamp=100.0, uptime_ms=5000.0)

        # Standard call
        std_out = self.engine.process_reading(rec)
        self.assertEqual(len(std_out), 47)

        # Diagnostic call
        prod_out, diag = self.engine.process_reading(rec, return_diagnostics=True)
        self.assertEqual(len(prod_out), 47)
        self.assertEqual(set(prod_out.keys()), set(PRODUCTION_OUTPUT_KEYS))
        self.assertIn("overall_health_score", diag)
        self.assertIn("formula", diag["overall_health_score"])


if __name__ == "__main__":
    unittest.main()
