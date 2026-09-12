import pandas as pd

def explain_latest(row: pd.Series) -> list[str]:
    factors = []
    if row.get("latency_trend", 0) > 0: factors.append(f"AI inference latency is rising by {row['latency_trend']:.2f} ms/min.")
    if row.get("cpu_trend", 0) > 0: factors.append(f"CPU utilization is rising by {row['cpu_trend']:.2f}%/min.")
    if row.get("memory_trend", 0) > 0: factors.append(f"Memory use is rising by {row['memory_trend']:.2f}%/min.")
    if row.get("deadline_rolling_mean", 0) > 0: factors.append("Deadline misses have appeared in the recent monitoring window.")
    return factors or ["No increasing degradation indicator is dominant in the current trailing window."]
