"""
Fault Detection, Root-Cause Analysis, Severity Classification, and Fault Counting.

Evaluates 8 multi-metric boolean fault flags, ranks candidate root causes across 11
internal diagnostic categories, determines fault severity, and counts active fault conditions.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from .features import DerivedFeatures
from .config import health_engineConfig


@dataclass
class FaultAnalysisResult:
    """Consolidated fault assessment."""
    cpu_overload_flag: bool
    memory_leak_flag: bool
    heap_exhaustion_flag: bool
    stack_risk_flag: bool
    deadline_miss_flag: bool
    task_starvation_flag: bool
    ai_latency_flag: bool
    queue_overflow_flag: bool
    fault_type: str
    fault_severity: str
    fault_count: int
    cause_contributions: Dict[str, float]


class FaultDetector:
    """
    Evaluates boolean fault conditions using multi-metric evidence,
    eliminating false alarms from single-sample instantaneous spikes.
    """

    def __init__(self, config: Optional[health_engineConfig] = None):
        self.config = config or health_engineConfig()

    def analyze_faults(
        self,
        reading: Dict[str, Any],
        features: DerivedFeatures,
        stats: Dict[str, Any],
        trends: Dict[str, str],
        growth_rates: Dict[str, Optional[float]],
        domain_scores: Dict[str, float],
    ) -> FaultAnalysisResult:
        """Run fault flag evaluation, root cause ranking, and severity classification."""

        # 1. CPU Overload Flag
        # Requires sustained high utilization over window or high saturation with degrading trend
        cpu_mean = stats.get("cpu_mean")
        curr_cpu = reading.get("cpu_utilization")
        cpu_thresh = self.config.cpu.overload_threshold_pct
        cpu_overload = False
        if cpu_mean is not None and cpu_mean >= cpu_thresh:
            cpu_overload = True
        elif curr_cpu is not None and curr_cpu >= cpu_thresh and (
            (cpu_mean is not None and cpu_mean >= 70.0) or trends.get("cpu_trend") == "DEGRADING"
        ):
            cpu_overload = True

        # 2. Memory Leak Flag
        # Requires persistent positive heap growth AND degrading trend over time
        heap_trend = trends.get("heap_trend")
        heap_growth = growth_rates.get("heap_growth_rate")
        free_heap = reading.get("free_heap")
        memory_leak = False
        if heap_trend == "DEGRADING" and heap_growth is not None and heap_growth > self.config.memory.leak_min_slope_pct_per_sec:
            memory_leak = True

        # 3. Heap Exhaustion Flag
        # Immediate critical shortage: heap utilization >= exhaustion_threshold OR minimum free heap < critical floor
        heap_util = reading.get("heap_utilization")
        min_free = reading.get("minimum_free_heap")
        heap_exhaustion = False
        if heap_util is not None and heap_util >= self.config.memory.heap_exhaustion_pct:
            heap_exhaustion = True
        elif min_free is not None and min_free <= self.config.memory.critical_min_free_heap_bytes:
            heap_exhaustion = True

        # 4. Stack Risk Flag
        stack_util = reading.get("stack_utilization")
        hwm = reading.get("stack_high_water_mark")
        stack_risk = False
        if stack_util is not None and stack_util >= self.config.stack.stack_risk_pct:
            stack_risk = True
        elif hwm is not None and hwm <= self.config.stack.critical_watermark_bytes:
            stack_risk = True

        # 5. Deadline Miss Flag
        miss_rate = stats.get("deadline_miss_rate", 0.0)
        curr_misses = reading.get("deadline_misses", 0)
        deadline_flag = False
        if miss_rate > self.config.timing.deadline_miss_rate_warning:
            deadline_flag = True
        elif curr_misses is not None and curr_misses > 0 and miss_rate > 0.0:
            deadline_flag = True

        # 6. Task Starvation Flag
        sched_delay = reading.get("scheduler_delay")
        task_state = str(reading.get("task_state", "")).upper()
        task_cpu = reading.get("task_cpu_utilization")
        task_starvation = False
        if sched_delay is not None and sched_delay >= self.config.timing.task_starvation_min_delay_ms:
            task_starvation = True
        elif task_state in ("READY", "BLOCKED") and task_cpu is not None and task_cpu < 1e-6 and curr_cpu is not None and curr_cpu > 85.0:
            task_starvation = True

        # 7. AI Latency Flag
        primary_lat = features.primary_latency_ms
        nominal_lat = self.config.ai.nominal_inference_time_ms
        lat_mean = stats.get("latency_mean")
        crit_lat = nominal_lat * self.config.ai.latency_critical_factor
        ai_latency = False
        if lat_mean is not None and lat_mean >= crit_lat:
            ai_latency = True
        elif primary_lat is not None and primary_lat >= crit_lat and trends.get("latency_trend") == "DEGRADING":
            ai_latency = True

        # 8. Queue Overflow Flag
        q_util = reading.get("queue_utilization")
        dropped = reading.get("dropped_messages")
        q_growth = growth_rates.get("queue_growth_rate")
        queue_overflow = False
        if q_util is not None and q_util >= self.config.queue.queue_overflow_pct:
            queue_overflow = True
        elif dropped is not None and dropped > 0 and q_util is not None and q_util > 80.0:
            queue_overflow = True

        # Tally active fault flags
        active_flags = [
            cpu_overload,
            memory_leak,
            heap_exhaustion,
            stack_risk,
            deadline_flag,
            task_starvation,
            ai_latency,
            queue_overflow,
        ]
        fault_count = sum(1 for f in active_flags if f)

        # Evidence-based Root Cause Analysis
        contributions = self._calculate_cause_contributions(
            reading=reading,
            features=features,
            stats=stats,
            trends=trends,
            growth_rates=growth_rates,
            domain_scores=domain_scores,
            flags={
                "cpu_overload": cpu_overload,
                "memory_leak": memory_leak,
                "heap_exhaustion": heap_exhaustion,
                "stack_risk": stack_risk,
                "deadline_miss": deadline_flag,
                "task_starvation": task_starvation,
                "ai_latency": ai_latency,
                "queue_overflow": queue_overflow,
            }
        )

        fault_type = self._diagnose_dominant_cause(contributions, fault_count)
        fault_severity = self._classify_severity(
            fault_count=fault_count,
            fault_type=fault_type,
            overall_health=domain_scores.get("overall_health_score", 100.0),
            critical_flags=[heap_exhaustion, stack_risk, deadline_flag]
        )

        return FaultAnalysisResult(
            cpu_overload_flag=cpu_overload,
            memory_leak_flag=memory_leak,
            heap_exhaustion_flag=heap_exhaustion,
            stack_risk_flag=stack_risk,
            deadline_miss_flag=deadline_flag,
            task_starvation_flag=task_starvation,
            ai_latency_flag=ai_latency,
            queue_overflow_flag=queue_overflow,
            fault_type=fault_type,
            fault_severity=fault_severity,
            fault_count=fault_count,
            cause_contributions=contributions,
        )

    def _calculate_cause_contributions(
        self,
        reading: Dict[str, Any],
        features: DerivedFeatures,
        stats: Dict[str, Any],
        trends: Dict[str, str],
        growth_rates: Dict[str, Optional[float]],
        domain_scores: Dict[str, float],
        flags: Dict[str, bool],
    ) -> Dict[str, float]:
        """Compute normalized evidence scores [0.0, 1.0] across all candidate causes."""
        c: Dict[str, float] = {}

        # CPU Overload
        c_score = 1.0 - (domain_scores.get("cpu_health_score", 100.0) / 100.0)
        if flags["cpu_overload"]:
            c_score += 0.5
        c["CPU_OVERLOAD"] = min(1.0, round(c_score, 3))

        # Memory Leak
        leak_score = 0.0
        if flags["memory_leak"]:
            leak_score += 0.65
        if trends.get("heap_trend") == "DEGRADING":
            leak_score += 0.25
        c["MEMORY_LEAK"] = min(1.0, round(leak_score, 3))

        # Heap Exhaustion
        heap_ex_score = 0.0
        if flags["heap_exhaustion"]:
            heap_ex_score += 0.75
        heap_util = reading.get("heap_utilization")
        if heap_util is not None and heap_util > 88.0:
            heap_ex_score += (heap_util - 88.0) * 0.02
        c["HEAP_EXHAUSTION"] = min(1.0, round(heap_ex_score, 3))

        # Stack Risk
        stack_score = 1.0 - (domain_scores.get("stack_health_score", 100.0) / 100.0)
        if flags["stack_risk"]:
            stack_score += 0.5
        c["STACK_RISK"] = min(1.0, round(stack_score, 3))

        # Task Starvation
        starv_score = 0.0
        if flags["task_starvation"]:
            starv_score += 0.7
        sched_del = reading.get("scheduler_delay")
        if sched_del is not None and sched_del > 15.0:
            starv_score += min(0.3, (sched_del - 15.0) * 0.02)
        c["TASK_STARVATION"] = min(1.0, round(starv_score, 3))

        # Scheduling Overload
        sched_ol = 1.0 - (domain_scores.get("task_health_score", 100.0) / 100.0)
        # NOTE: stats.get("cpu_mean", 0) previously ignored its default whenever
        # "cpu_mean" was present-but-None (i.e. cpu_utilization never observed in
        # the window), raising "'>' not supported between NoneType and int".
        # Guard explicitly instead of relying on dict.get's default.
        cpu_mean_val = stats.get("cpu_mean")
        if reading.get("context_switches") is not None and cpu_mean_val is not None and cpu_mean_val > 80:
            sched_ol += 0.2
        c["SCHEDULING_OVERLOAD"] = min(1.0, round(sched_ol, 3))

        # Deadline Violation
        dl_score = 1.0 - (domain_scores.get("timing_health_score", 100.0) / 100.0)
        if flags["deadline_miss"]:
            dl_score += 0.5
        c["DEADLINE_VIOLATION"] = min(1.0, round(dl_score, 3))

        # AI Latency / AI Overload
        ai_score = 1.0 - (domain_scores.get("ai_health_score", 100.0) / 100.0)
        if flags["ai_latency"]:
            ai_score += 0.5
        c["AI_LATENCY"] = min(1.0, round(ai_score, 3))

        # Queue Congestion / Communication Overload
        q_score = 0.0
        if flags["queue_overflow"]:
            q_score += 0.6
        q_util = reading.get("queue_utilization")
        if q_util is not None and q_util > 80.0:
            q_score += (q_util - 80.0) * 0.02
        c["QUEUE_CONGESTION"] = min(1.0, round(q_score, 3))

        # Thermal Stress
        temp = reading.get("system_temperature")
        t_score = 0.0
        if temp is not None and temp > self.config.hardware.temp_warning_celsius:
            t_score = min(1.0, (temp - self.config.hardware.temp_warning_celsius) / 20.0)
        c["THERMAL_STRESS"] = min(1.0, round(t_score, 3))

        # Hardware/System Instability (Watchdog / Resets)
        instab = 0.0
        wd = reading.get("watchdog_resets")
        if wd is not None and wd > 0:
            instab += 0.8
        c["SYSTEM_INSTABILITY"] = min(1.0, round(instab, 3))

        return c

    def _diagnose_dominant_cause(self, contributions: Dict[str, float], fault_count: int) -> str:
        """Select dominant supported cause or return MULTIPLE_FAULTS / NONE."""
        significant_causes = [(k, v) for k, v in contributions.items() if v >= 0.50]

        if not significant_causes:
            # Check for lower-level warning
            sub_causes = [(k, v) for k, v in contributions.items() if v >= 0.35]
            if not sub_causes:
                return "NONE"
            sub_causes.sort(key=lambda x: x[1], reverse=True)
            return sub_causes[0][0]

        # If 2 or more distinct root causes have high simultaneous evidence
        if len(significant_causes) >= 2:
            significant_causes.sort(key=lambda x: x[1], reverse=True)
            top1_k, top1_v = significant_causes[0]
            top2_k, top2_v = significant_causes[1]
            # If top two are close in score and multiple flags are active
            if (top1_v - top2_v) < 0.20 and fault_count >= 2:
                return "MULTIPLE_FAULTS"

        significant_causes.sort(key=lambda x: x[1], reverse=True)
        return significant_causes[0][0]

    def _classify_severity(
        self,
        fault_count: int,
        fault_type: str,
        overall_health: float,
        critical_flags: List[bool]
    ) -> str:
        """
        Classify fault severity: NONE, LOW, MEDIUM, HIGH, CRITICAL.
        Reflects magnitude, persistence, and safety implications.
        """
        # Critical conditions
        if any(critical_flags) or overall_health <= 25.0:
            return "CRITICAL"
        if fault_count >= 3:
            return "CRITICAL"

        # High severity
        # NOTE: the "named dangerous category" clause used to fire on its own,
        # with no requirement that any fault flag was actually active. That let
        # a merely-dominant (but sub-threshold, "significant"-tier ~0.35-0.5
        # evidence) root cause label force fault_severity=HIGH even when
        # fault_count==0 and overall_health was in the 90s - contradicting the
        # rest of the output. It's now gated on fault_count >= 1 so it can only
        # escalate an already-real, currently-active fault to HIGH.
        if (
            fault_count >= 2
            or overall_health <= 50.0
            or (fault_count >= 1 and fault_type in ("HEAP_EXHAUSTION", "STACK_RISK", "CPU_OVERLOAD", "MEMORY_LEAK"))
        ):
            if fault_type != "NONE":
                return "HIGH"

        # Medium severity
        if fault_count == 1 or overall_health <= 70.0:
            return "MEDIUM"

        # Low severity
        if overall_health < 85.0 or fault_type != "NONE":
            return "LOW"

        return "NONE"
