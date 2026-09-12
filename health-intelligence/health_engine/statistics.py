"""
Statistical Analysis, Trend Analysis, and Growth Rate Engine for Person 2.

Implements robust moving window statistics, OLS regression slope trends,
normalized growth rates, and rigorous handling of insufficient history.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from .history import HistoryManager


class StatisticsEngine:
    """
    Computes statistical moments, OLS regression trends, and growth rates
    over irregular time-series intervals from the rolling history buffer.
    """

    def __init__(self, config=None):
        self.config = config

    def compute_summary_stats(self, values: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Compute (mean, std, min, max) for a numeric series.
        - If len == 0: (None, None, None, None)
        - If len == 1: (val, 0.0, val, val)
        - If len >= 2: sample mean and sample standard deviation (Bessel's correction N-1).
        """
        if not values:
            return None, None, None, None

        n = len(values)
        if n == 1:
            val = float(values[0])
            return round(val, 3), 0.0, round(val, 3), round(val, 3)

        mean_val = sum(values) / n
        variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        std_val = math.sqrt(max(0.0, variance))
        min_val = min(values)
        max_val = max(values)

        return round(mean_val, 3), round(std_val, 3), round(min_val, 3), round(max_val, 3)

    def compute_ols_slope(self, time_series: List[Tuple[float, float]]) -> Optional[float]:
        """
        Compute Ordinary Least Squares (OLS) slope m = dy/dt over time in seconds.
        Returns None if points < min_trend_points or time span < 0.5s.
        """
        min_points = self.config.window.min_trend_points if self.config else 5
        if len(time_series) < min_points:
            return None

        t_vals = [pt[0] for pt in time_series]
        y_vals = [pt[1] for pt in time_series]

        t_span = t_vals[-1] - t_vals[0]
        if t_span < 0.5: # Too short a duration to measure a temporal rate
            return None

        n = len(time_series)
        t_mean = sum(t_vals) / n
        y_mean = sum(y_vals) / n

        numerator = sum((t - t_mean) * (y - y_mean) for t, y in zip(t_vals, y_vals))
        denominator = sum((t - t_mean) ** 2 for t in t_vals)

        if denominator < 1e-8:
            return 0.0

        return numerator / denominator

    def classify_trend(
        self,
        time_series: List[Tuple[float, float]],
        degrading_threshold: float,
        improving_threshold: float,
        higher_is_degrading: bool = True
    ) -> str:
        """
        Classify temporal trend into: 'IMPROVING', 'STABLE', 'DEGRADING', or 'INSUFFICIENT_DATA'.
        """
        slope = self.compute_ols_slope(time_series)
        if slope is None:
            return "INSUFFICIENT_DATA"

        if higher_is_degrading:
            if slope > degrading_threshold:
                return "DEGRADING"
            elif slope < improving_threshold:
                return "IMPROVING"
            else:
                return "STABLE"
        else:
            # Lower is degrading (e.g. free heap, headroom)
            if slope < improving_threshold:
                return "DEGRADING"
            elif slope > degrading_threshold:
                return "IMPROVING"
            else:
                return "STABLE"

    def compute_growth_rate(
        self,
        time_series: List[Tuple[float, float]],
    ) -> Optional[float]:
        """
        Compute growth rate in percent per second (%/sec) using normalized linear slope:
        growth_rate = (dy/dt) / (|y_mean| + epsilon) * 100%.
        Consistent sign convention:
        Positive (+) = metric value is increasing over time.
        Negative (-) = metric value is decreasing over time.
        Returns None when insufficient history.
        """
        slope = self.compute_ols_slope(time_series)
        if slope is None:
            return None

        y_vals = [pt[1] for pt in time_series]
        y_mean = sum(y_vals) / len(y_vals)

        eps = 1e-4
        denom = abs(y_mean) if abs(y_mean) > eps else 1.0
        pct_rate = (slope / denom) * 100.0
        return round(pct_rate, 4)

    def calculate_deadline_miss_rate(self, history: HistoryManager) -> float:
        """
        Calculate deadline miss rate over the rolling window.
        Misses per execution: delta_misses / delta_executions (if execution count available).
        Fallback: delta_misses / delta_t (misses/sec).
        """
        miss_series = history.get_time_series("deadline_misses")
        if not miss_series:
            return 0.0

        if len(miss_series) < 2:
            return 0.0

        delta_misses = max(0, miss_series[-1][1] - miss_series[0][1])
        exec_series = history.get_time_series("task_execution_count")

        if len(exec_series) >= 2:
            delta_exec = max(0, exec_series[-1][1] - exec_series[0][1])
            if delta_exec > 0:
                return round(delta_misses / delta_exec, 5)

        # Fallback to misses per second
        time_delta = max(0.1, miss_series[-1][0] - miss_series[0][0])
        return round(delta_misses / time_delta, 5)
