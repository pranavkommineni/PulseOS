import pandas as pd

def recommendations(row: pd.Series, threshold: float, rsul_minutes: float) -> list[str]:
    result = []
    if row.get("latency_trend", 0) > 0 or row.get("ai_latency_ms", 0) >= threshold * .8:
        result.append("Optimize or reload the AI model; consider reducing inference frequency.")
    if row.get("cpu_percent", 0) >= 70 or row.get("cpu_trend", 0) > .3:
        result.append("Reduce competing workload and reserve CPU time for perception/control tasks.")
    if row.get("memory_percent", 0) >= 65 or row.get("memory_trend", 0) > .2:
        result.append("Inspect memory allocation and restart the affected service during a safe maintenance window.")
    if row.get("deadline_rolling_mean", 0) > 0:
        result.append("Review FreeRTOS task priorities, queue contention, and scheduling intervals.")
    if rsul_minutes < 20:
        result.append("Perform preventive maintenance before the predicted critical point.")
    return result or ["Continue monitoring; no immediate predictive maintenance action is indicated."]
