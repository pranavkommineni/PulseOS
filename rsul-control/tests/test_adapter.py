import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rsul_control.adapter.person2_adapter import adapt


def test_legacy_adapter():
    x = adapt({
        "timestamp": 220,
        "shi": 57,
        "health_state": "DEGRADING",
        "degradation_rate": -3.4,
        "degradation_acceleration": 0.8,
        "root_cause": "MEMORY_PRESSURE",
        "cause_confidence": 0.81,
    })
    assert x.mode == "legacy"
    assert x.payload["shi"] == 57


def test_reject_unknown_contract():
    try:
        adapt({"timestamp": 1})
    except ValueError:
        return
    raise AssertionError("Unsupported payload should be rejected")
