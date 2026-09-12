"""
Degradation Analysis and Health State Classification Engine for Person 2.

Calculates the degradation_index (0.0 to 1.0), computes the degradation_rate (dD/dt)
and health_change_rate (dH/dt), and classifies the system into 5 health states.
"""

from typing import List, Tuple, Optional, Dict, Any
from .config import health_engineConfig


class DegradationAnalyzer:
    """
    Computes degradation index, temporal degradation rate, health change rate,
    and classifies health states with configurable boundaries.
    """

    def __init__(self, config: Optional[health_engineConfig] = None):
        self.config = config or health_engineConfig()
        self.degradation_history: List[Tuple[float, float]] = []
        self.health_history: List[Tuple[float, float]] = []

    def reset(self) -> None:
        """Reset historical tracking on device reboot."""
        self.degradation_history.clear()
        self.health_history.clear()

    def compute_degradation_index(
        self,
        overall_health: float,
        fault_count: int,
        fault_severity: str,
        trends: Dict[str, str]
    ) -> float:
        """
        Compute degradation_index on a normalized scale:
        0.0 = no meaningful degradation
        1.0 = complete system degradation / failure.
        """
        # Base degradation from health deficit
        health_deficit = max(0.0, (100.0 - overall_health) / 100.0)

        # Fault count penalty
        fault_penalty = min(0.30, fault_count * 0.04)

        # Severity weighting
        sev_penalty = {
            "NONE": 0.0,
            "LOW": 0.02,
            "MEDIUM": 0.05,
            "HIGH": 0.10,
            "CRITICAL": 0.20,
        }.get(fault_severity, 0.0)

        # Persistent trend degradation penalty
        degrading_trends_count = sum(1 for t in trends.values() if t == "DEGRADING")
        trend_penalty = min(0.15, degrading_trends_count * 0.03)

        raw_index = health_deficit * 0.75 + fault_penalty + sev_penalty + trend_penalty
        return max(0.0, min(1.0, round(raw_index, 4)))

    def update_and_calculate_rates(
        self,
        current_ts: float,
        degradation_index: float,
        overall_health: float,
        min_trend_points: int = 5
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Record current values and compute:
        1. degradation_rate (dD/dt): rate of degradation index change per second.
           Positive (+) = degradation worsening.
        2. health_change_rate (dH/dt): rate of overall health score change per second.
           Positive (+) = health improving.
        """
        self.degradation_history.append((current_ts, degradation_index))
        self.health_history.append((current_ts, overall_health))

        # Prune old history (keep last 120s / 300 points)
        max_pts = self.config.window.max_history_points if self.config else 300
        window_sec = self.config.window.window_duration_sec if self.config else 120.0

        while len(self.degradation_history) > max_pts:
            self.degradation_history.pop(0)
            self.health_history.pop(0)

        cutoff = current_ts - window_sec
        while self.degradation_history and self.degradation_history[0][0] < cutoff:
            self.degradation_history.pop(0)
            self.health_history.pop(0)

        # If insufficient points
        if len(self.degradation_history) < min_trend_points:
            return None, None

        time_span = self.degradation_history[-1][0] - self.degradation_history[0][0]
        if time_span < 0.5:
            return None, None

        deg_rate = self._compute_slope(self.degradation_history)
        health_rate = self._compute_slope(self.health_history)

        return (
            round(deg_rate, 5) if deg_rate is not None else None,
            round(health_rate, 4) if health_rate is not None else None,
        )

    def classify_health_state(
        self,
        overall_health: float,
        degradation_index: float,
        fault_severity: str,
        fault_count: int,
        degradation_rate: Optional[float]
    ) -> str:
        """
        Classify system into: HEALTHY, STRESSED, DEGRADING, CRITICAL, FAILURE_RISK.
        """
        t = self.config.health_states

        # Condition for FAILURE_RISK: imminent crash or multiple critical failures
        if overall_health < t.critical_min_score or degradation_index >= t.critical_max_degradation:
            return "FAILURE_RISK"
        if fault_severity == "CRITICAL" and fault_count >= 2:
            return "FAILURE_RISK"

        # Condition for CRITICAL: active single critical fault or very low health
        if overall_health < t.degrading_min_score or degradation_index >= t.degrading_max_degradation:
            return "CRITICAL"
        if fault_severity == "CRITICAL":
            return "CRITICAL"

        # Condition for DEGRADING: worsening trend or moderate degradation
        is_worsening = degradation_rate is not None and degradation_rate > 0.005
        if is_worsening or overall_health < t.stressed_min_score or degradation_index >= t.stressed_max_degradation:
            return "DEGRADING"

        # Condition for STRESSED: elevated utilization or warnings without full degradation
        if overall_health < t.healthy_min_score or degradation_index >= t.healthy_max_degradation or fault_count > 0:
            return "STRESSED"

        # Otherwise: HEALTHY
        return "HEALTHY"

    def _compute_slope(self, series: List[Tuple[float, float]]) -> Optional[float]:
        """Compute OLS linear slope."""
        n = len(series)
        t_vals = [pt[0] for pt in series]
        y_vals = [pt[1] for pt in series]

        t_mean = sum(t_vals) / n
        y_mean = sum(y_vals) / n

        num = sum((t - t_mean) * (y - y_mean) for t, y in zip(t_vals, y_vals))
        den = sum((t - t_mean) ** 2 for t in t_vals)

        if den < 1e-8:
            return 0.0
        return num / den
