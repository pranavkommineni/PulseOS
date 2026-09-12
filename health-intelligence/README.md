# Person 2: Real-time Health Intelligence System for RTOS-based Edge-AI

A mathematically rigorous, stream-oriented health monitoring and predictive intelligence engine for RTOS Edge-AI systems.

Person 2 ingests raw runtime telemetry from **Person 1** (44 metrics), cleans and reconciles corrupted or incomplete readings, maintains persistent rolling historical state, computes OLS regression trends and growth rates, evaluates 0–100 domain health scores, detects active faults, diagnoses root causes across 17 categories, and streams strictly validated health intelligence (47 metrics) to **Person 3**.

---

## Directory Structure

```
Embed/
├── health_engine/                     # Core Person-2 Package
│   ├── __init__.py              # Package entry point & version
│   ├── config.py                # Centralized calibratable parameters & scenario profiles
│   ├── validator.py             # Input validation, range checks, and consistency reconciliation
│   ├── history.py               # Rolling window buffer & device reboot/reset detection
│   ├── features.py              # Internal derived features avoiding double counting
│   ├── statistics.py            # Moving statistics, OLS regression slope trends, growth rates
│   ├── health.py                # 0-100 Multi-domain health scores & bottleneck aggregation
│   ├── faults.py                # 8 Multi-metric fault flags & evidence-based root cause diagnosis
│   ├── degradation.py           # Degradation index, degradation rate, health change rate, health states
│   ├── engine.py                # Stateful stream processor & strict 47-key output JSON contract
│   ├── diagnostics.py           # Non-intrusive mathematical explainability & trace engine
│   └── synthetic.py             # Correlated multivariate telemetry generator (11 scenarios)
├── tests/                       # Complete Automated Test Suite (27 Scenarios + Contract)
│   ├── __init__.py
│   ├── test_validation.py       # Tests 1-10: Boundary, missing, null, inconsistent values
│   ├── test_temporal.py         # Tests 11-16: Timestamps, device restarts, startup, history depth
│   ├── test_scenarios.py        # Tests 17-27: Healthy, degradation, faults, recovery
│   ├── test_contract.py         # JSON schema contract, field invariance, NaN/Inf checks
│   └── run_all_tests.py         # Master test runner script
├── docs/                        # Complete Documentation
│   ├── METRIC_SPECIFICATION.md  # Exhaustive 44-input and 47-output specification tables
│   └── ARCHITECTURE.md          # Architectural rationale, mathematical formulations, calibration
├── examples/                    # Working Demonstrations & JSON Samples
│   ├── demo_stream.py           # Real-time streaming simulation script
│   ├── sample_input.json        # Authoritative Person 1 input JSON record
│   └── sample_output.json       # Authoritative Person 2 output JSON record
├── requirements.txt             # Minimal dependencies
└── README.md                    # This document
```

---

## Quickstart & Stream Processing API

```python
from health_engine.engine import HealthIntelligenceEngine

# 1. Initialize the stateful engine
engine = HealthIntelligenceEngine()

# 2. Process incoming telemetry records (dict or JSON string)
telemetry_reading = {
    "timestamp": 100.0,
    "sample_id": 1,
    "uptime_ms": 5000.0,
    "scenario_id": "1",
    "cpu_utilization": 34.5,
    "cpu_idle": 65.5,
    # ... all other Person 1 fields
}

# 3. Stream processing
output_json = engine.process_reading(telemetry_reading)

print(f"Overall Health: {output_json['overall_health_score']:.1f}/100")
print(f"Health State:   {output_json['health_state']}")
print(f"Fault Type:     {output_json['fault_type']}")
print(f"Degradation:    {output_json['degradation_index']:.3f}")
```

### Resetting Between Runs or Scenarios
```python
engine.reset()
```

### Optional Diagnostic & Explainability Mode
The production output contract is never modified. Querying explanations is fully decoupled:
```python
output, diagnostics = engine.process_reading(telemetry_reading, return_diagnostics=True)
print(diagnostics["overall_health_score"]["reason"])
print(diagnostics["fault_type"]["formula"])
```

---

## Running the Complete Test Suite

The test suite covers all 27 specified test cases + contract invariance:

```bash
python tests/run_all_tests.py
```

Or using standard unittest:
```bash
python -m unittest discover tests
```

---

## Key Mathematical & Architectural Highlights

1. **Strict Authoritative Interfaces**:
   - Ingests exactly the 44 Person 1 metrics specified in Section 2.
   - Emits exactly the 47 Person 2 metrics specified in Section 2 & 34.
   - Never outputs `NaN`, `Infinity`, or `-Infinity`.
2. **Missing vs Observed Zero**:
   - `None` is never coerced to zero. Missingness triggers deterministic derivation when mathematically possible (e.g. `cpu_util = 100 - idle`) or incurs a sample confidence penalty.
3. **Continuous OLS Trends & Normalized Growth Rates**:
   - Trends are computed via Ordinary Least Squares linear regression slope over irregular time intervals, categorized into `IMPROVING`, `STABLE`, `DEGRADING`, or `INSUFFICIENT_DATA`.
   - Growth rates are normalized (%/sec) with a universal sign convention (positive = physical metric value increasing).
4. **Bottleneck-Constrained Health Aggregation**:
   - Domain scores (0–100) are aggregated using scenario weights modified by a non-linear bottleneck constraint to prevent single-point failures from being masked.
5. **Distinct Degradation & Health Rates**:
   - `degradation_index`: $[0.0, 1.0]$ composite degradation metric.
   - `degradation_rate`: $dD/dt$ ($1/\text{sec}$), positive when degradation is worsening.
   - `health_change_rate`: $dH/dt$ ($\text{points}/\text{sec}$), positive when health is improving.
6. **Correlated Synthetic Data Generator**:
   - Realistically simulates all 11 operational scenarios: `HEALTHY`, `CPU_OVERLOAD`, `MEMORY_LEAK`, `HEAP_EXHAUSTION`, `STACK_RISK`, `DEADLINE_DEGRADATION`, `TASK_STARVATION`, `AI_LATENCY`, `QUEUE_CONGESTION`, `MULTIPLE_FAULTS`, and `RECOVERY`.
