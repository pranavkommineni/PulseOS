"""
Input Validation, Cleaning, and Consistency Reconciler for Person 2.

Enforces physical boundaries, mathematical invariants, distinguishes Missing from Observed Zero,
and derives values deterministically when mathematically justified.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List


# Exact 44 input metrics specified in Person 1 -> Person 2 interface
REQUIRED_INPUT_METRICS = [
    "timestamp",
    "sample_id",
    "uptime_ms",
    "scenario_id",
    "cpu_utilization",
    "cpu_idle",
    "task_cpu_utilization",
    "total_heap",
    "free_heap",
    "used_heap",
    "heap_utilization",
    "minimum_free_heap",
    "stack_high_water_mark",
    "stack_utilization",
    "task_name",
    "task_priority",
    "task_state",
    "task_execution_time",
    "task_period",
    "task_jitter",
    "task_execution_count",
    "deadline_misses",
    "context_switches",
    "task_switches",
    "scheduler_delay",
    "active_task_count",
    "inference_time",
    "min_inference_time",
    "max_inference_time",
    "average_inference_time",
    "inference_count",
    "inference_frequency",
    "queue_length",
    "queue_capacity",
    "queue_utilization",
    "messages_sent",
    "messages_received",
    "dropped_messages",
    "interrupt_count",
    "interrupt_latency",
    "power_consumption",
    "system_temperature",
    "watchdog_resets",
    "system_resets",
]


@dataclass
class CleanedReading:
    """Represents a validated, cleaned, and reconciled telemetry record."""
    data: Dict[str, Any]
    confidence: float = 1.0
    anomalies: List[str] = field(default_factory=list)
    is_valid: bool = True


class InputValidator:
    """
    Validates, filters impossible values, reconciles inconsistencies,
    and fills derivable fields mathematically.
    """

    def __init__(self, config=None):
        self.config = config

    def validate_and_clean(self, raw_data: Dict[str, Any]) -> CleanedReading:
        """
        Process raw input dictionary:
        1. Extract expected fields.
        2. Type-cast and range check.
        3. Reconcile mathematical invariants.
        4. Calculate sample confidence.
        """
        cleaned: Dict[str, Any] = {}
        anomalies: List[str] = []
        confidence_penalties = 0.0

        # Step 1: Base Extraction & Numeric Sanitization
        for metric in REQUIRED_INPUT_METRICS:
            val = raw_data.get(metric, None)
            if val is None:
                cleaned[metric] = None
            else:
                cleaned[metric] = self._sanitize_scalar(metric, val, anomalies)

        # Fundamental requirement checks: timestamp or uptime_ms must exist
        ts = cleaned.get("timestamp")
        uptime = cleaned.get("uptime_ms")
        if ts is None and uptime is None:
            anomalies.append("MISSING_TEMPORAL_ANCHOR: Neither timestamp nor uptime_ms provided")
            return CleanedReading(data=cleaned, confidence=0.0, anomalies=anomalies, is_valid=False)

        # Synthesize timestamp from uptime_ms or vice versa if one is missing
        if ts is None and uptime is not None:
            cleaned["timestamp"] = float(uptime) / 1000.0
            anomalies.append("IMPUTED_TIMESTAMP_FROM_UPTIME")
        elif uptime is None and ts is not None:
            cleaned["uptime_ms"] = float(ts) * 1000.0
            anomalies.append("IMPUTED_UPTIME_FROM_TIMESTAMP")

        # Step 2: CPU Domain Consistency
        cpu_util = cleaned.get("cpu_utilization")
        cpu_idle = cleaned.get("cpu_idle")
        task_cpu = cleaned.get("task_cpu_utilization")

        # Range validation: [0, 100]
        if cpu_util is not None and not (0.0 <= cpu_util <= 100.0):
            anomalies.append(f"IMPOSSIBLE_CPU_UTILIZATION: {cpu_util}")
            cleaned["cpu_utilization"] = None
            cpu_util = None
            confidence_penalties += 0.2

        if cpu_idle is not None and not (0.0 <= cpu_idle <= 100.0):
            anomalies.append(f"IMPOSSIBLE_CPU_IDLE: {cpu_idle}")
            cleaned["cpu_idle"] = None
            cpu_idle = None
            confidence_penalties += 0.1

        if task_cpu is not None and not (0.0 <= task_cpu <= 100.0):
            anomalies.append(f"IMPOSSIBLE_TASK_CPU: {task_cpu}")
            cleaned["task_cpu_utilization"] = None
            task_cpu = None

        # Derivation & Invariant: cpu_utilization + cpu_idle == 100
        if cpu_util is None and cpu_idle is not None:
            cleaned["cpu_utilization"] = round(100.0 - cpu_idle, 3)
            cpu_util = cleaned["cpu_utilization"]
        elif cpu_idle is None and cpu_util is not None:
            cleaned["cpu_idle"] = round(100.0 - cpu_util, 3)
            cpu_idle = cleaned["cpu_idle"]
        elif cpu_util is not None and cpu_idle is not None:
            cpu_sum = cpu_util + cpu_idle
            if abs(cpu_sum - 100.0) > 4.0: # Significant inconsistency
                anomalies.append(f"CPU_INCONSISTENCY: cpu_util ({cpu_util}) + cpu_idle ({cpu_idle}) = {cpu_sum}")
                confidence_penalties += 0.15
                # Reconcile: trust cpu_utilization as primary signal, clamp idle
                cleaned["cpu_idle"] = max(0.0, min(100.0, round(100.0 - cpu_util, 3)))

        # Step 3: Memory & Heap Domain Consistency
        total_h = cleaned.get("total_heap")
        used_h = cleaned.get("used_heap")
        free_h = cleaned.get("free_heap")
        heap_util = cleaned.get("heap_utilization")
        min_free_h = cleaned.get("minimum_free_heap")

        # Range checks: non-negative
        for h_metric, h_val in [("total_heap", total_h), ("used_heap", used_h), ("free_heap", free_h), ("minimum_free_heap", min_free_h)]:
            if h_val is not None and h_val < 0:
                anomalies.append(f"NEGATIVE_MEMORY_VALUE: {h_metric}={h_val}")
                cleaned[h_metric] = None
                confidence_penalties += 0.15

        # Re-fetch after negativity check
        total_h = cleaned.get("total_heap")
        used_h = cleaned.get("used_heap")
        free_h = cleaned.get("free_heap")

        # Derivations when 2 of 3 (total, used, free) are present
        if total_h is None and used_h is not None and free_h is not None:
            total_h = used_h + free_h
            cleaned["total_heap"] = total_h
        elif used_h is None and total_h is not None and free_h is not None:
            used_h = max(0, total_h - free_h)
            cleaned["used_heap"] = used_h
        elif free_h is None and total_h is not None and used_h is not None:
            free_h = max(0, total_h - used_h)
            cleaned["free_heap"] = free_h

        # Invariant: used + free ≈ total
        if total_h is not None and used_h is not None and free_h is not None:
            if total_h > 0:
                calc_sum = used_h + free_h
                if abs(calc_sum - total_h) > max(1024, total_h * 0.05):
                    anomalies.append(f"HEAP_INCONSISTENCY: used ({used_h}) + free ({free_h}) != total ({total_h})")
                    confidence_penalties += 0.2
                    # Reconcile: trust total and used if used <= total
                    if used_h <= total_h:
                        cleaned["free_heap"] = total_h - used_h
                        free_h = cleaned["free_heap"]
                    else:
                        cleaned["used_heap"] = total_h
                        cleaned["free_heap"] = 0
                        used_h = total_h
                        free_h = 0

        # Heap utilization derivation / validation
        if heap_util is not None and not (0.0 <= heap_util <= 100.0):
            anomalies.append(f"IMPOSSIBLE_HEAP_UTILIZATION: {heap_util}")
            cleaned["heap_utilization"] = None
            heap_util = None
            confidence_penalties += 0.1

        if heap_util is None and total_h is not None and total_h > 0 and used_h is not None:
            cleaned["heap_utilization"] = round((used_h / total_h) * 100.0, 3)
        elif heap_util is not None and total_h is not None and total_h > 0 and used_h is not None:
            derived_util = (used_h / total_h) * 100.0
            if abs(heap_util - derived_util) > 3.0:
                anomalies.append(f"HEAP_UTILIZATION_MISMATCH: reported {heap_util}% vs calculated {derived_util:.2f}%")
                confidence_penalties += 0.1
                # Trust calculated ratio
                cleaned["heap_utilization"] = round(derived_util, 3)

        # Minimum free heap check: cannot exceed free_heap or total_heap
        if min_free_h is not None:
            if free_h is not None and min_free_h > free_h:
                anomalies.append(f"MINIMUM_FREE_HEAP_EXCEEDS_FREE: min_free={min_free_h} > free={free_h}")
                # Minimum free heap watermark must be at least as low as current free heap
                cleaned["minimum_free_heap"] = free_h
            if total_h is not None and min_free_h > total_h:
                cleaned["minimum_free_heap"] = min(total_h, free_h if free_h is not None else total_h)

        # Step 4: Stack Domain Consistency
        stack_util = cleaned.get("stack_utilization")
        if stack_util is not None and not (0.0 <= stack_util <= 100.0):
            anomalies.append(f"IMPOSSIBLE_STACK_UTILIZATION: {stack_util}")
            cleaned["stack_utilization"] = None
            confidence_penalties += 0.1

        stack_hwm = cleaned.get("stack_high_water_mark")
        if stack_hwm is not None and stack_hwm < 0:
            anomalies.append(f"NEGATIVE_STACK_HWM: {stack_hwm}")
            cleaned["stack_high_water_mark"] = None
            confidence_penalties += 0.1

        # Step 5: Queue Domain Consistency
        q_len = cleaned.get("queue_length")
        q_cap = cleaned.get("queue_capacity")
        q_util = cleaned.get("queue_utilization")

        if q_len is not None and q_len < 0:
            anomalies.append(f"NEGATIVE_QUEUE_LENGTH: {q_len}")
            cleaned["queue_length"] = None
            q_len = None
            confidence_penalties += 0.1

        if q_cap is not None and q_cap <= 0:
            anomalies.append(f"NON_POSITIVE_QUEUE_CAPACITY: {q_cap}")
            cleaned["queue_capacity"] = None
            q_cap = None
            confidence_penalties += 0.1

        if q_len is not None and q_cap is not None:
            if q_len > q_cap:
                anomalies.append(f"QUEUE_LENGTH_EXCEEDS_CAPACITY: len={q_len} > cap={q_cap}")
                confidence_penalties += 0.15
                # Cap utilization at 100% or overflow representation
                cleaned["queue_utilization"] = 100.0
            else:
                calc_q_util = (q_len / q_cap) * 100.0
                if q_util is None:
                    cleaned["queue_utilization"] = round(calc_q_util, 3)
                elif abs(q_util - calc_q_util) > 5.0:
                    anomalies.append(f"QUEUE_UTILIZATION_MISMATCH: reported {q_util}% vs calc {calc_q_util:.2f}%")
                    cleaned["queue_utilization"] = round(calc_q_util, 3)

        # Step 6: Inference Timing Consistency
        inf_t = cleaned.get("inference_time")
        min_inf = cleaned.get("min_inference_time")
        max_inf = cleaned.get("max_inference_time")
        avg_inf = cleaned.get("average_inference_time")

        # Non-negative checks
        for i_name, i_val in [("inference_time", inf_t), ("min_inference_time", min_inf), ("max_inference_time", max_inf), ("average_inference_time", avg_inf)]:
            if i_val is not None and i_val < 0:
                anomalies.append(f"NEGATIVE_INFERENCE_TIME: {i_name}={i_val}")
                cleaned[i_name] = None
                confidence_penalties += 0.1

        # Re-fetch
        min_inf = cleaned.get("min_inference_time")
        max_inf = cleaned.get("max_inference_time")
        avg_inf = cleaned.get("average_inference_time")
        inf_t = cleaned.get("inference_time")

        # Reconcile min <= avg <= max
        if min_inf is not None and max_inf is not None:
            if min_inf > max_inf:
                anomalies.append(f"INFERENCE_BOUND_INVERTED: min={min_inf} > max={max_inf}")
                # Swap to fix inversion
                cleaned["min_inference_time"], cleaned["max_inference_time"] = max_inf, min_inf
                min_inf, max_inf = cleaned["min_inference_time"], cleaned["max_inference_time"]
                confidence_penalties += 0.1

        if avg_inf is not None and min_inf is not None and max_inf is not None:
            if not (min_inf <= avg_inf <= max_inf):
                anomalies.append(f"AVG_INFERENCE_OUTSIDE_BOUNDS: avg={avg_inf} not in [{min_inf}, {max_inf}]")
                confidence_penalties += 0.1
                # Clamp average to bounds
                cleaned["average_inference_time"] = max(min_inf, min(max_inf, avg_inf))
        elif avg_inf is None and inf_t is not None:
            cleaned["average_inference_time"] = inf_t

        # Step 7: Timing & Task Counters
        for count_metric in ["deadline_misses", "context_switches", "task_switches", "inference_count",
                             "messages_sent", "messages_received", "dropped_messages", "interrupt_count",
                             "watchdog_resets", "system_resets", "task_execution_count"]:
            val = cleaned.get(count_metric)
            if val is not None and val < 0:
                anomalies.append(f"NEGATIVE_COUNTER: {count_metric}={val}")
                cleaned[count_metric] = None
                confidence_penalties += 0.1

        for delay_metric in ["scheduler_delay", "task_jitter", "task_execution_time", "task_period", "interrupt_latency"]:
            val = cleaned.get(delay_metric)
            if val is not None and val < 0:
                anomalies.append(f"NEGATIVE_TIMING: {delay_metric}={val}")
                cleaned[delay_metric] = None
                confidence_penalties += 0.1

        # Temperature realistic bounds [-40C, +150C]
        temp = cleaned.get("system_temperature")
        if temp is not None and not (-40.0 <= temp <= 150.0):
            anomalies.append(f"IMPOSSIBLE_TEMPERATURE: {temp}")
            cleaned["system_temperature"] = None
            confidence_penalties += 0.1

        # Power consumption non-negative
        power = cleaned.get("power_consumption")
        if power is not None and power < 0:
            anomalies.append(f"NEGATIVE_POWER: {power}")
            cleaned["power_consumption"] = None

        final_confidence = max(0.0, min(1.0, round(1.0 - confidence_penalties, 2)))
        return CleanedReading(data=cleaned, confidence=final_confidence, anomalies=anomalies, is_valid=True)

    def _sanitize_scalar(self, metric: str, val: Any, anomalies: List[str]) -> Any:
        """Convert input value to correct native type; reject malformed or NaN values."""
        if val is None:
            return None

        # String-based metrics
        if metric in ("task_name", "task_state", "scenario_id"):
            return str(val).strip()

        # Integer-based metrics
        if metric in ("sample_id", "total_heap", "free_heap", "used_heap", "minimum_free_heap",
                      "stack_high_water_mark", "task_priority", "task_execution_count",
                      "deadline_misses", "context_switches", "task_switches", "active_task_count",
                      "inference_count", "queue_length", "queue_capacity", "messages_sent",
                      "messages_received", "dropped_messages", "interrupt_count",
                      "watchdog_resets", "system_resets"):
            try:
                if isinstance(val, bool): # Python bool is a subclass of int
                    anomalies.append(f"TYPE_ERROR_BOOL_FOR_INT: {metric}={val}")
                    return None
                f_val = float(val)
                if math.isnan(f_val) or math.isinf(f_val):
                    anomalies.append(f"NAN_OR_INF_VALUE: {metric}={val}")
                    return None
                return int(f_val)
            except (ValueError, TypeError):
                anomalies.append(f"TYPE_CONVERSION_ERROR: {metric}={val}")
                return None

        # Floating-point metrics
        try:
            if isinstance(val, bool):
                anomalies.append(f"TYPE_ERROR_BOOL_FOR_FLOAT: {metric}={val}")
                return None
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                anomalies.append(f"NAN_OR_INF_VALUE: {metric}={val}")
                return None
            return round(f_val, 4)
        except (ValueError, TypeError):
            anomalies.append(f"TYPE_CONVERSION_ERROR: {metric}={val}")
            return None
