"""
Demonstration Stream for Person 2 Health Intelligence Engine.

Simulates a real-time streaming telemetry sequence from Person 1,
processes each record through Person 2, and outputs the production JSON for Person 3.
Also produces examples/sample_input.json and examples/sample_output.json.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from health_engine.engine import HealthIntelligenceEngine
from health_engine.synthetic import SyntheticTelemetryGenerator


def run_demo():
    engine = HealthIntelligenceEngine()
    synth = SyntheticTelemetryGenerator(seed=777)

    # Generate a sequence showing healthy baseline transitioning into CPU overload
    healthy_recs = synth.generate_scenario("HEALTHY", num_samples=5, start_time=100.0)
    cpu_recs = synth.generate_scenario("CPU_OVERLOAD", num_samples=10, start_time=105.0)
    all_recs = healthy_recs + cpu_recs

    examples_dir = Path(__file__).resolve().parent
    examples_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PERSON-2 REAL-TIME HEALTH INTELLIGENCE STREAM DEMO")
    print("=" * 75)

    last_input = None
    last_output = None

    for i, rec in enumerate(all_recs):
        # Process record through Person 2
        output, diagnostics = engine.process_reading(rec, return_diagnostics=True)

        last_input = rec
        last_output = output

        print(
            f"Step {i+1:02d} | Time: {output['timestamp']:6.1f}s | "
            f"CPU: {rec['cpu_utilization']:5.1f}% | "
            f"Health: {output['overall_health_score']:5.1f} | "
            f"State: {output['health_state']:<10} | "
            f"Fault: {output['fault_type']:<15} | "
            f"Sev: {output['fault_severity']:<8} | "
            f"Degradation: {output['degradation_index']:.3f}"
        )

    # Save sample_input.json and sample_output.json
    input_path = examples_dir / "sample_input.json"
    output_path = examples_dir / "sample_output.json"

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(last_input, f, indent=2)
    print(f"\n[+] Saved sample input record to: {input_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(last_output, f, indent=2)
    print(f"[+] Saved sample output record to: {output_path}")

    print("\n--- DIAGNOSTIC TRACE SAMPLE (Non-intrusive Explainability) ---")
    print("Overall Health Score Trace:")
    print(json.dumps(diagnostics["overall_health_score"], indent=2))
    print("\nDominant Fault Diagnosis Trace:")
    print(json.dumps(diagnostics["fault_type"], indent=2))
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
