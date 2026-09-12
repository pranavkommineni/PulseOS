"""
Feature Engineering Layer for Person 2 Health Intelligence Engine.

Computes internal derived signals (headrooms, saturation ratios, scheduling pressures)
with mathematical justification, avoiding metric duplication or correlated inflation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class DerivedFeatures:
    """Internal feature representation used for health scores and fault detection."""
    # CPU
    cpu_headroom_pct: Optional[float] = None
    task_cpu_ratio: Optional[float] = None

    # Memory / Heap
    heap_headroom_pct: Optional[float] = None
    free_heap_ratio: Optional[float] = None
    min_free_heap_ratio: Optional[float] = None

    # Stack
    stack_headroom_pct: Optional[float] = None

    # Scheduling / Timing
    execution_to_period_ratio: Optional[float] = None
    scheduler_delay_ratio: Optional[float] = None

    # Latency / AI
    primary_latency_ms: Optional[float] = None
    ai_latency_pressure: Optional[float] = None

    # Queue & Communication
    queue_pressure_ratio: Optional[float] = None
    dropped_message_ratio: Optional[float] = None

    # Hardware / Resources
    thermal_margin_celsius: Optional[float] = None
    power_pressure_ratio: Optional[float] = None


class FeatureEngineer:
    """
    Extracts physically and logically defensible internal features from cleaned readings.
    """

    def __init__(self, config=None):
        self.config = config

    def extract_features(self, reading: Dict[str, Any]) -> DerivedFeatures:
        """Derive internal features for current reading."""
        feat = DerivedFeatures()
        eps = 1e-6

        # 1. CPU Features
        cpu_util = reading.get("cpu_utilization")
        if cpu_util is not None:
            feat.cpu_headroom_pct = max(0.0, 100.0 - cpu_util)

        task_cpu = reading.get("task_cpu_utilization")
        if task_cpu is not None and cpu_util is not None:
            feat.task_cpu_ratio = min(1.0, task_cpu / max(cpu_util, 1.0))

        # 2. Memory / Heap Features
        heap_util = reading.get("heap_utilization")
        total_heap = reading.get("total_heap")
        free_heap = reading.get("free_heap")
        min_free_heap = reading.get("minimum_free_heap")

        if heap_util is not None:
            feat.heap_headroom_pct = max(0.0, 100.0 - heap_util)

        if total_heap is not None and total_heap > 0:
            if free_heap is not None:
                feat.free_heap_ratio = min(1.0, max(0.0, free_heap / total_heap))
            if min_free_heap is not None:
                feat.min_free_heap_ratio = min(1.0, max(0.0, min_free_heap / total_heap))

        # 3. Stack Features
        stack_util = reading.get("stack_utilization")
        if stack_util is not None:
            feat.stack_headroom_pct = max(0.0, 100.0 - stack_util)

        # 4. Scheduling / Timing Features
        exec_t = reading.get("task_execution_time")
        period = reading.get("task_period")
        sched_delay = reading.get("scheduler_delay")

        if exec_t is not None and period is not None and period > 0:
            feat.execution_to_period_ratio = round(exec_t / period, 4)

        if sched_delay is not None and period is not None and period > 0:
            feat.scheduler_delay_ratio = round(sched_delay / period, 4)

        # 5. AI Latency Features
        # Primary latency signal selection: average_inference_time is primary smoothed signal;
        # fallback to instantaneous inference_time if average not provided.
        avg_inf = reading.get("average_inference_time")
        inf_t = reading.get("inference_time")
        primary_lat = avg_inf if avg_inf is not None else inf_t
        feat.primary_latency_ms = primary_lat

        nominal_lat = self.config.ai.nominal_inference_time_ms if self.config else 35.0
        if primary_lat is not None and nominal_lat > 0:
            feat.ai_latency_pressure = round(primary_lat / nominal_lat, 3)

        # 6. Queue & Communication Features
        q_len = reading.get("queue_length")
        q_cap = reading.get("queue_capacity")
        if q_len is not None and q_cap is not None and q_cap > 0:
            feat.queue_pressure_ratio = min(1.0, q_len / q_cap)

        dropped = reading.get("dropped_messages")
        sent = reading.get("messages_sent")
        if dropped is not None and sent is not None:
            total_msgs = sent + dropped
            if total_msgs > 0:
                feat.dropped_message_ratio = min(1.0, dropped / total_msgs)

        # 7. Hardware Resources
        temp = reading.get("system_temperature")
        crit_temp = self.config.hardware.temp_critical_celsius if self.config else 85.0
        if temp is not None:
            feat.thermal_margin_celsius = max(0.0, crit_temp - temp)

        power = reading.get("power_consumption")
        crit_power = self.config.hardware.power_critical_mw if self.config else 3500.0
        if power is not None and crit_power > 0:
            feat.power_pressure_ratio = min(2.0, power / crit_power)

        return feat
