from __future__ import annotations

PERSON2_COLUMNS = [
    "timestamp", "cpu_mean", "cpu_std", "cpu_min", "cpu_max", "cpu_trend", "cpu_growth_rate",
    "memory_mean", "memory_std", "memory_trend", "memory_growth_rate", "heap_trend", "heap_growth_rate",
    "stack_trend", "stack_growth_rate", "latency_mean", "latency_max", "latency_trend", "latency_growth_rate",
    "task_delay_trend", "deadline_miss_rate", "deadline_trend", "queue_trend", "queue_growth_rate",
    "cpu_health_score", "memory_health_score", "stack_health_score", "task_health_score", "timing_health_score",
    "ai_health_score", "resource_health_score", "overall_health_score", "health_state", "cpu_overload_flag",
    "memory_leak_flag", "heap_exhaustion_flag", "stack_risk_flag", "deadline_miss_flag", "task_starvation_flag",
    "ai_latency_flag", "queue_overflow_flag", "fault_type", "fault_severity", "fault_count", "degradation_index",
    "degradation_rate", "health_change_rate",
]

NUMERIC_COLUMNS = [c for c in PERSON2_COLUMNS if c not in {"timestamp", "health_state", "fault_type"}]


def validate_person2(df):
    missing = [c for c in PERSON2_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Person-2 input is missing {len(missing)} required columns: {missing}")
    # Positional contract: keep exactly the agreed order at the interface.
    return df.loc[:, PERSON2_COLUMNS].copy()
