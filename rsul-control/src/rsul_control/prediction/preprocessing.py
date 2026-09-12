"""Non-destructive telemetry cleaning for synthetic or future RTOS CSV data."""
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ["timestamp", "ai_latency_ms", "cpu_percent", "memory_percent", "health_score"]
NUMERIC = ["runtime_minutes", "cpu_percent", "memory_percent", "free_heap_kb", "stack_usage_bytes",
           "ai_latency_ms", "task_execution_time_ms", "deadline_misses", "task_jitter_ms",
           "context_switches_per_sec", "queue_utilization_percent", "dropped_messages", "health_score"]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Telemetry is missing required columns: {missing}")
    clean = df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
    clean = clean.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    if "runtime_minutes" not in clean:
        clean["runtime_minutes"] = (clean["timestamp"] - clean["timestamp"].iloc[0]).dt.total_seconds() / 60
    for col in [c for c in NUMERIC if c in clean]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean[col] = clean[col].interpolate().bfill().ffill()
        # Runtime is the independent time axis, so never clip it.
        if col == "runtime_minutes":
            continue
        # Winsorise only extreme measurement noise; preserves realistic degradation.
        lo, hi = clean[col].quantile([0.01, 0.99])
        if hi > lo:
            clean[col] = clean[col].clip(lo, hi)
    return clean.reset_index(drop=True)

def preprocess_csv(source: str | Path, destination: str | Path) -> pd.DataFrame:
    out = preprocess(pd.read_csv(source))
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False)
    return out
