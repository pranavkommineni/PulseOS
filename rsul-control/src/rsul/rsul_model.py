from dataclasses import dataclass

import numpy as np


@dataclass
class RsulResult:
    current_is_critical: bool
    predicted_critical_runtime_minutes: float | None
    time_to_critical_minutes: float


def time_to_critical(
    current_latency: float,
    latency_slope_per_min: float,
    current_runtime_minutes: float,
    threshold_ms: float,
) -> RsulResult:
    """Estimate threshold-crossing time from the current latency trajectory."""
    if current_latency >= threshold_ms:
        return RsulResult(True, current_runtime_minutes, 0.0)
    if latency_slope_per_min <= 0:
        return RsulResult(False, None, float("inf"))
    remaining = max(0.0, (threshold_ms - current_latency) / latency_slope_per_min)
    return RsulResult(False, current_runtime_minutes + remaining, remaining)


def rsul_from_model_prediction(
    current_latency: float,
    predicted_latency: float,
    current_runtime_minutes: float,
    horizon_minutes: int,
    threshold_ms: float,
) -> RsulResult:
    """Estimate RSUL using the ML forecast over its trained prediction horizon."""
    if current_latency >= threshold_ms:
        return RsulResult(True, current_runtime_minutes, 0.0)

    horizon = max(float(horizon_minutes), 1.0)
    slope = (float(predicted_latency) - float(current_latency)) / horizon
    return time_to_critical(current_latency, slope, current_runtime_minutes, threshold_ms)


def forecast_line(
    current_runtime_minutes: float,
    current_latency: float,
    predicted_latency: float,
    horizon_minutes: int = 10,
    points: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a transparent linear forecast line between now and the model horizon."""
    runtime = np.linspace(
        current_runtime_minutes,
        current_runtime_minutes + max(horizon_minutes, 1),
        max(points, 2),
    )
    fraction = (runtime - current_runtime_minutes) / max(float(horizon_minutes), 1.0)
    forecast = current_latency + fraction * (predicted_latency - current_latency)
    return runtime, forecast


def crossing_from_forecast(
    runtime: np.ndarray, forecast: np.ndarray, threshold_ms: float
) -> float | None:
    hit = np.flatnonzero(forecast >= threshold_ms)
    return float(runtime[hit[0]]) if len(hit) else None
