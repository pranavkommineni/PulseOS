import numpy as np

def prototype_failure_risk(current_latency: float, threshold_ms: float, rsul_minutes: float,
                           degradation_rate: float, deadline_misses: float) -> float:
    """Transparent demo-risk mapping, NOT a calibrated real-world probability."""
    proximity = np.clip(current_latency / threshold_ms, 0, 1.2) * 45
    urgency = 35 if rsul_minutes <= 0 else 30 * np.exp(-max(rsul_minutes, 0) / 15)
    trend = np.clip(degradation_rate, 0, 5) * 4
    history = min(deadline_misses, 5) * 4
    return float(np.clip(proximity + urgency + trend + history, 0, 100))
