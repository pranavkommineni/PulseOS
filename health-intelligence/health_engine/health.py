"""
Multi-Domain Health Scoring Engine for Person 2.

Calculates normalized 0–100 health scores (100 = best, 0 = worst) across 7 domains
and aggregates them into an overall health score using context-aware scenario weights
and a mathematically justified bottleneck penalty function.
"""

import logging
import math
from typing import Dict, Any, Optional
from .features import DerivedFeatures
from .config import health_engineConfig, ScenarioProfile

logger = logging.getLogger(__name__)


class HealthScoreEngine:
    """
    Computes domain health scores:
    - cpu_health_score
    - memory_health_score
    - stack_health_score
    - task_health_score
    - timing_health_score
    - ai_health_score
    - resource_health_score
    - overall_health_score
    """

    def __init__(self, config: Optional[health_engineConfig] = None):
        self.config = config or health_engineConfig()

    def calculate_domain_scores(
        self,
        reading: Dict[str, Any],
        features: DerivedFeatures,
        stats: Dict[str, Any],
        trends: Dict[str, str],
        growth_rates: Dict[str, Optional[float]],
    ) -> Dict[str, float]:
        """Compute all 8 health scores normalized to [0.0, 100.0]."""

        # 1. CPU Health Score
        cpu_score = self._compute_cpu_health(reading, features, stats, trends)

        # 2. Memory Health Score
        memory_score = self._compute_memory_health(
            reading, features, stats, trends, growth_rates
        )

        # 3. Stack Health Score
        stack_score = self._compute_stack_health(reading, features, stats, trends)

        # 4. Task Health Score
        task_score = self._compute_task_health(reading, features, stats, trends)

        # 5. Timing Health Score
        timing_score = self._compute_timing_health(reading, features, stats, trends)

        # 6. AI Health Score
        ai_score = self._compute_ai_health(reading, features, stats, trends)

        # 7. Resource Health Score
        resource_score = self._compute_resource_health(reading, features, stats, trends)

        # 8. Overall Health Score with Context-Aware Weighting and Bottleneck Penalty
        scenario_id = reading.get("scenario_id")
        profile = self.config.get_scenario_profile(scenario_id)

        domain_scores = {
            "cpu_health_score": cpu_score,
            "memory_health_score": memory_score,
            "stack_health_score": stack_score,
            "task_health_score": task_score,
            "timing_health_score": timing_score,
            "ai_health_score": ai_score,
            "resource_health_score": resource_score,
        }

        overall_score = self._aggregate_overall_health(domain_scores, profile)
        domain_scores["overall_health_score"] = overall_score

        return domain_scores

    def _compute_cpu_health(
        self,
        reading: Dict[str, Any],
        features: DerivedFeatures,
        stats: Dict[str, Any],
        trends: Dict[str, str],
    ) -> float:
        """Compute CPU domain health (0-100)."""
        cpu_util = reading.get("cpu_utilization")
        if cpu_util is None:
            # Neutral default when missing
            return 85.0

        healthy_limit = self.config.cpu.healthy_upper_pct  # e.g. 70%
        overload_limit = self.config.cpu.overload_threshold_pct  # e.g. 88%

        if cpu_util <= healthy_limit:
            base_score = 100.0 - (cpu_util / healthy_limit) * 10.0  # 100 down to 90
        elif cpu_util <= overload_limit:
            ratio = (cpu_util - healthy_limit) / (overload_limit - healthy_limit)
            base_score = 90.0 - ratio * 40.0  # 90 down to 50
        else:
            ratio = min(1.0, (cpu_util - overload_limit) / (100.0 - overload_limit))
            base_score = 50.0 - ratio * 45.0  # 50 down to 5

        # Penalties for variability and sustained degradation
        cpu_std = stats.get("cpu_std")
        if cpu_std is not None and cpu_std > 15.0:
            base_score -= min(10.0, (cpu_std - 15.0) * 0.5)

        if trends.get("cpu_trend") == "DEGRADING":
            base_score -= 8.0
        elif trends.get("cpu_trend") == "IMPROVING":
            base_score += 3.0

        return max(0.0, min(100.0, round(base_score, 2)))

    def _compute_memory_health(
        self,
        reading: Dict[str, Any],
        features: DerivedFeatures,
        stats: Dict[str, Any],
        trends: Dict[str, str],
        growth_rates: Dict[str, Optional[float]],
    ) -> float:
        """Compute Memory (heap) domain health (0-100)."""

        # ---------------------------------------------------------
        # 1. Get heap utilization
        # ---------------------------------------------------------
        heap_util = reading.get("heap_utilization")

        # If heap_utilization is missing, derive it from total_heap and free_heap.
        if heap_util is None:
            total_heap = reading.get("total_heap")
            free_heap = reading.get("free_heap")

            try:
                if (
                    total_heap is not None
                    and free_heap is not None
                    and float(total_heap) > 0
                    and float(free_heap) >= 0
                ):
                    total_heap = float(total_heap)
                    free_heap = float(free_heap)

                    used_heap = max(0.0, total_heap - free_heap)
                    heap_util = (used_heap / total_heap) * 100.0
                else:
                    # No reliable memory information available.
                    return 85.0

            except (TypeError, ValueError, ZeroDivisionError):
                return 85.0

        # ---------------------------------------------------------
        # 2. Validate heap utilization
        # ---------------------------------------------------------
        try:
            heap_util = float(heap_util)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid heap_utilization=%r. Using fallback memory health score.",
                heap_util,
            )
            return 85.0

        # NaN/inf would slip past the range check below and poison the score.
        if not math.isfinite(heap_util):
            logger.warning(
                "Non-finite heap_utilization=%r. Using fallback memory health score.",
                heap_util,
            )
            return 85.0

        # Detect impossible percentage values.
        if heap_util < 0.0 or heap_util > 100.0:
            logger.warning(
                "heap_utilization=%.3f%% is outside the valid 0-100%% range. "
                "Clamping to valid range.",
                heap_util,
            )
            heap_util = max(0.0, min(100.0, heap_util))

        # ---------------------------------------------------------
        # 3. Base health score from heap utilization
        # ---------------------------------------------------------
        healthy_limit = float(self.config.memory.heap_healthy_upper_pct)  # e.g. 75%
        exhaustion_limit = float(self.config.memory.heap_exhaustion_pct)  # e.g. 92%

        # Protect against invalid configuration.
        healthy_limit = max(0.0, min(100.0, healthy_limit))
        exhaustion_limit = max(healthy_limit, min(100.0, exhaustion_limit))

        if heap_util <= healthy_limit:
            # 0% -> 100, healthy_limit -> 90
            if healthy_limit > 0:
                base_score = 100.0 - (heap_util / healthy_limit) * 10.0
            else:
                base_score = 100.0

        elif heap_util <= exhaustion_limit:
            # healthy_limit -> 90, exhaustion_limit -> 40
            denominator = exhaustion_limit - healthy_limit
            ratio = (
                (heap_util - healthy_limit) / denominator if denominator > 0 else 1.0
            )
            base_score = 90.0 - ratio * 50.0

        else:
            # exhaustion_limit -> 40, 100% -> 0
            denominator = 100.0 - exhaustion_limit
            ratio = (
                (heap_util - exhaustion_limit) / denominator if denominator > 0 else 1.0
            )
            ratio = max(0.0, min(1.0, ratio))
            base_score = 40.0 - ratio * 40.0

        # ---------------------------------------------------------
        # 4. Minimum free heap penalty
        # ---------------------------------------------------------
        min_free = reading.get("minimum_free_heap")

        try:
            if min_free is not None:
                min_free = float(min_free)
                crit_free = float(self.config.memory.critical_min_free_heap_bytes)

                if crit_free > 0 and min_free < crit_free:
                    deficit_ratio = 1.0 - (min_free / crit_free)
                    deficit_ratio = max(0.0, min(1.0, deficit_ratio))
                    base_score -= deficit_ratio * 25.0

        except (TypeError, ValueError):
            logger.warning(
                "Invalid minimum_free_heap=%r. Skipping minimum-free-heap penalty.",
                min_free,
            )

        # ---------------------------------------------------------
        # 5. Heap degradation trend penalty
        # ---------------------------------------------------------
        if trends.get("heap_trend") == "DEGRADING":
            base_score -= 12.0

        # ---------------------------------------------------------
        # 6. Heap growth-rate (leak) penalty
        # ---------------------------------------------------------
        rate = growth_rates.get("heap_growth_rate")

        try:
            if rate is not None:
                rate = float(rate)
                threshold = float(self.config.memory.leak_min_slope_pct_per_sec)

                # Only positive growth beyond the configured leak threshold is penalized.
                if rate > threshold:
                    excess_rate = rate - threshold
                    base_score -= min(15.0, excess_rate * 10.0)

        except (TypeError, ValueError):
            logger.warning(
                "Invalid heap_growth_rate=%r. Skipping leak-growth penalty.", rate
            )

        # ---------------------------------------------------------
        # 7. Final clamp
        # ---------------------------------------------------------
        return round(max(0.0, min(100.0, base_score)), 2)

    def _compute_stack_health(
        self,
        reading: Dict[str, Any],
        features: DerivedFeatures,
        stats: Dict[str, Any],
        trends: Dict[str, str],
    ) -> float:
        """Compute Stack domain health (0-100)."""
        stack_util = reading.get("stack_utilization")
        if stack_util is None:
            return 90.0

        healthy_limit = self.config.stack.stack_healthy_upper_pct  # 75%
        risk_limit = self.config.stack.stack_risk_pct  # 88%

        if stack_util <= healthy_limit:
            base_score = 100.0 - (stack_util / healthy_limit) * 10.0
        elif stack_util <= risk_limit:
            ratio = (stack_util - healthy_limit) / (risk_limit - healthy_limit)
            base_score = 90.0 - ratio * 45.0
        else:
            ratio = min(1.0, (stack_util - risk_limit) / (100.0 - risk_limit))
            base_score = 45.0 - ratio * 45.0

        hwm = reading.get("stack_high_water_mark")
        crit_hwm = self.config.stack.critical_watermark_bytes
        if hwm is not None and hwm < crit_hwm:
            base_score -= 25.0

        if trends.get("stack_trend") == "DEGRADING":
            base_score -= 10.0

        return max(0.0, min(100.0, round(base_score, 2)))

    def _compute_task_health(
        self,
        reading: Dict[str, Any],
        features: DerivedFeatures,
        stats: Dict[str, Any],
        trends: Dict[str, str],
    ) -> float:
        """Compute Task / Scheduling domain health (0-100)."""
        sched_delay = reading.get("scheduler_delay")
        jitter = reading.get("task_jitter")

        score = 100.0
        max_delay = self.config.timing.max_acceptable_scheduler_delay_ms  # 15ms
        max_jitter = self.config.timing.max_acceptable_jitter_ms  # 10ms

        if sched_delay is not None and sched_delay > 0:
            if sched_delay > max_delay:
                score -= min(
                    50.0, ((sched_delay - max_delay) / max_delay) * 40.0 + 15.0
                )
            else:
                score -= (sched_delay / max_delay) * 12.0

        if jitter is not None and jitter > 0:
            if jitter > max_jitter:
                score -= min(30.0, ((jitter - max_jitter) / max_jitter) * 20.0 + 10.0)
            else:
                score -= (jitter / max_jitter) * 8.0

        if trends.get("task_delay_trend") == "DEGRADING":
            score -= 10.0

        return max(0.0, min(100.0, round(score, 2)))

    def _compute_timing_health(
        self,
        reading: Dict[str, Any],
        features: DerivedFeatures,
        stats: Dict[str, Any],
        trends: Dict[str, str],
    ) -> float:
        """Compute Timing & Deadline domain health (0-100)."""
        score = 100.0
        miss_rate = stats.get("deadline_miss_rate", 0.0)
        warn_rate = self.config.timing.deadline_miss_rate_warning  # 0.005
        crit_rate = self.config.timing.deadline_miss_rate_critical  # 0.02

        if miss_rate > 0:
            if miss_rate <= warn_rate:
                score -= (miss_rate / warn_rate) * 20.0  # 100 down to 80
            elif miss_rate <= crit_rate:
                ratio = (miss_rate - warn_rate) / (crit_rate - warn_rate)
                score = 80.0 - ratio * 50.0  # 80 down to 30
            else:
                score = max(0.0, 30.0 - (miss_rate / crit_rate) * 20.0)

        # WCET execution ratio penalty
        exec_ratio = features.execution_to_period_ratio
        if exec_ratio is not None and exec_ratio > 0.85:
            score -= min(25.0, (exec_ratio - 0.85) * 100.0)

        if trends.get("deadline_trend") == "DEGRADING":
            score -= 15.0

        return max(0.0, min(100.0, round(score, 2)))

    def _compute_ai_health(
        self,
        reading: Dict[str, Any],
        features: DerivedFeatures,
        stats: Dict[str, Any],
        trends: Dict[str, str],
    ) -> float:
        """Compute Edge-AI Inference domain health (0-100)."""
        score = 100.0
        primary_lat = features.primary_latency_ms
        nominal_lat = self.config.ai.nominal_inference_time_ms  # 35ms

        if primary_lat is not None and nominal_lat > 0:
            ratio = primary_lat / nominal_lat
            warn_factor = self.config.ai.latency_warning_factor  # 1.35
            crit_factor = self.config.ai.latency_critical_factor  # 1.80

            if ratio <= 1.05:
                score = 100.0
            elif ratio <= warn_factor:
                score = 100.0 - ((ratio - 1.05) / (warn_factor - 1.05)) * 25.0
            elif ratio <= crit_factor:
                r = (ratio - warn_factor) / (crit_factor - warn_factor)
                score = 75.0 - r * 45.0
            else:
                r = min(2.0, (ratio - crit_factor) / crit_factor)
                score = max(0.0, 30.0 - r * 30.0)

        if trends.get("latency_trend") == "DEGRADING":
            score -= 12.0
        elif trends.get("latency_trend") == "IMPROVING":
            score += 4.0

        return max(0.0, min(100.0, round(score, 2)))

    def _compute_resource_health(
        self,
        reading: Dict[str, Any],
        features: DerivedFeatures,
        stats: Dict[str, Any],
        trends: Dict[str, str],
    ) -> float:
        """Compute Resource (Queue, Hardware, Power, Thermal) domain health (0-100)."""
        score = 100.0

        # Queue pressure
        q_util = reading.get("queue_utilization")
        if q_util is not None:
            if q_util > self.config.queue.queue_warning_pct:
                score -= min(30.0, (q_util - self.config.queue.queue_warning_pct) * 2.0)

        # Dropped messages
        dropped_ratio = features.dropped_message_ratio
        if dropped_ratio is not None and dropped_ratio > 0:
            score -= min(40.0, dropped_ratio * 200.0)

        # Thermal stress
        temp = reading.get("system_temperature")
        if temp is not None:
            if temp > self.config.hardware.temp_warning_celsius:
                crit = self.config.hardware.temp_critical_celsius
                warn = self.config.hardware.temp_warning_celsius
                temp_pen = min(35.0, ((temp - warn) / max(1.0, crit - warn)) * 35.0)
                score -= temp_pen

        # Interrupt latency
        int_lat = reading.get("interrupt_latency")
        if (
            int_lat is not None
            and int_lat > self.config.hardware.interrupt_latency_critical_ms
        ):
            score -= 15.0

        if trends.get("queue_trend") == "DEGRADING":
            score -= 8.0

        return max(0.0, min(100.0, round(score, 2)))

    def _aggregate_overall_health(
        self, domain_scores: Dict[str, float], profile: ScenarioProfile
    ) -> float:
        """
        Aggregate domain health scores using scenario-configured weights
        with a non-linear bottleneck constraint preventing masked single-point failures.
        """
        w_sum = (
            profile.weight_cpu
            + profile.weight_memory
            + profile.weight_stack
            + profile.weight_task
            + profile.weight_timing
            + profile.weight_ai
            + profile.weight_resource
        )
        if w_sum <= 0:
            w_sum = 1.0

        weighted_avg = (
            domain_scores["cpu_health_score"] * profile.weight_cpu
            + domain_scores["memory_health_score"] * profile.weight_memory
            + domain_scores["stack_health_score"] * profile.weight_stack
            + domain_scores["task_health_score"] * profile.weight_task
            + domain_scores["timing_health_score"] * profile.weight_timing
            + domain_scores["ai_health_score"] * profile.weight_ai
            + domain_scores["resource_health_score"] * profile.weight_resource
        ) / w_sum

        # Bottleneck principle: In an RTOS, if one critical domain collapses (e.g. Memory=5%),
        # the overall system cannot claim to be 85% healthy.
        min_domain = min(domain_scores.values())

        if min_domain < 40.0:
            # Constrain overall health by the weakest link
            constrained = min_domain + 0.35 * (weighted_avg - min_domain)
            final_score = min(weighted_avg, constrained)
        else:
            final_score = weighted_avg

        return max(0.0, min(100.0, round(final_score, 2)))
