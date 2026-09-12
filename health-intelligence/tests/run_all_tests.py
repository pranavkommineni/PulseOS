"""
Master Test Runner for Person 2 Health Intelligence System.

Discovers and executes all test suites, verifying all 27 specified test cases
and output contract requirements.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_all_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Discover and add all tests from the tests directory
    tests_dir = Path(__file__).resolve().parent
    discovered = loader.discover(start_dir=str(tests_dir), pattern="test_*.py")
    suite.addTests(discovered)

    print("=" * 70)
    print("PERSON-2 HEALTH INTELLIGENCE SYSTEM: AUTOMATED TEST SUITE")
    print("=" * 70)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("ALL TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("SOME TESTS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
