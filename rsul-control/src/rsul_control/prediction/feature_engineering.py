import pandas as pd
from .config import SETTINGS

def make_features(df: pd.DataFrame, window: int = SETTINGS.rolling_window) -> pd.DataFrame:
    """Trailing-only features: never use future rows, preventing temporal leakage."""
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    for col, prefix in [("ai_latency_ms", "latency"), ("cpu_percent", "cpu"),
                        ("memory_percent", "memory"), ("health_score", "health"),
                        ("task_execution_time_ms", "execution"), ("deadline_misses", "deadline")]:
        if col in out:
            out[f"{prefix}_rolling_mean"] = out[col].rolling(window, min_periods=1).mean()
            out[f"{prefix}_trend"] = out[col].diff().rolling(window, min_periods=1).mean().fillna(0)
    out["degradation_rate"] = (-out.get("health_trend", 0)).clip(lower=0) + out.get("latency_trend", 0).clip(lower=0) / 5
    return out

def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "health_state", "critical_indicator", "future_ai_latency_ms", "future_health_score"}
    return [c for c in df.select_dtypes("number").columns if c not in excluded]
