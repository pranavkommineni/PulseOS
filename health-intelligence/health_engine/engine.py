"""
Person 2: Real-time Health Intelligence System Engine.

Core coordinator implementing the stream processing interface:
process_reading(input_json) -> output_json

Enforces the strict 47-key output JSON contract, maintains state between invocations,
manages reboot resets, and provides non-intrusive diagnostic capabilities.
"""

import json
import math
from typing import Dict, Any, Optional, Union, Tuple

from .config import health_engineConfig
from .validator import InputValidator, CleanedReading
from .history import HistoryManager
from .features import FeatureEngineer, DerivedFeatures
from .statistics import StatisticsEngine
from .health import HealthScoreEngine
from .faults import FaultDetector, FaultAnalysisResult
from .degradation import DegradationAnalyzer
from .diagnostics import DiagnosticsManager, MetricExplanation


# Exact 47 output keys specified in Person 2 -> Person 3 contract
PRODUCTION_OUTPUT_KEYS = [
    "timestamp",
    "cpu_mean",
    "cpu_std",
    "cpu_min",
    "cpu_max",
    "cpu_trend",
    "cpu_growth_rate",
    "memory_mean",
    "memory_std",
    "memory_trend",
    "memory_growth_rate",
    "heap_trend",
    "heap_growth_rate",
    "stack_trend",
    "stack_growth_rate",
    "latency_mean",
    "latency_max",
    "latency_trend",
    "latency_growth_rate",
    "task_delay_trend",
    "deadline_miss_rate",
    "deadline_trend",
    "queue_trend",
    "queue_growth_rate",
    "cpu_health_score",
    "memory_health_score",
    "stack_health_score",
    "task_health_score",
    "timing_health_score",
    "ai_health_score",
    "resource_health_score",
    "overall_health_score",
    "health_state",
    "cpu_overload_flag",
    "memory_leak_flag",
    "heap_exhaustion_flag",
    "stack_risk_flag",
    "deadline_miss_flag",
    "task_starvation_flag",
    "ai_latency_flag",
    "queue_overflow_flag",
    "fault_type",
    "fault_severity",
    "fault_count",
    "degradation_index",
    "degradation_rate",
    "health_change_rate",
]


class HealthIntelligenceEngine:
    """
    Stateful streaming health intelligence processor for RTOS-based Edge-AI systems.
    """

    def __init__(self, config: Optional[health_engineConfig] = None):
        self.config = config or health_engineConfig()
        self.validator = InputValidator(self.config)
        self.history = HistoryManager(self.config)
        self.features = FeatureEngineer(self.config)
        self.statistics = StatisticsEngine(self.config)
        self.health = HealthScoreEngine(self.config)
        self.faults = FaultDetector(self.config)
        self.degradation = DegradationAnalyzer(self.config)
        self.diagnostics = DiagnosticsManager()

    def reset(self) -> None:
        """Reset internal streaming state for a new device run or scenario."""
        self.history.reset_state()
        self.degradation.reset()
        self.diagnostics = DiagnosticsManager()

    def process_reading(
        self,
        input_data: Union[str, Dict[str, Any]],
        return_diagnostics: bool = False
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Process a single runtime telemetry reading from Person 1 and return health intelligence for Person 3.

        Parameters:
            input_data: JSON string or Python dictionary matching the 44 Person 1 metrics.
            return_diagnostics: If True, returns a tuple (production_json, diagnostics_dict).
                                If False (default), returns strictly the production JSON dictionary.

        Returns:
            Dictionary matching the exact 47 Person 2 output fields.
        """
        # Step 0: Parse JSON string if necessary
        if isinstance(input_data, str):
            try:
                raw_dict = json.loads(input_data)
            except Exception:
                raw_dict = {}
        elif isinstance(input_data, dict):
            raw_dict = dict(input_data)
        else:
            raw_dict = {}

        # Step 1: Input Validation, Cleaning, and Consistency Reconciling
        cleaned_reading: CleanedReading = self.validator.validate_and_clean(raw_dict)
        cleaned_data = cleaned_reading.data

        # Step 1b: Reject unrecoverable input (e.g. no timestamp AND no uptime_ms).
        # Without a temporal anchor nothing downstream (history, trends, degradation
        # rates) can be computed safely, so we short-circuit with a valid-but-inert
        # 47-key contract response instead of letting a None timestamp/mean crash
        # the pipeline further down.
        if not cleaned_reading.is_valid:
            fallback = self._build_invalid_input_output(cleaned_reading)
            if return_diagnostics:
                return fallback, self.diagnostics.get_all_explanations()
            return fallback

        # Step 2: History & Reset Management
        is_reset, reset_reason = self.history.add_reading(
            cleaned_data=cleaned_data,
            confidence=cleaned_reading.confidence
        )
        if is_reset:
            self.degradation.reset()

        # Step 3: Feature Engineering
        derived_feat = self.features.extract_features(cleaned_data)

        # Step 4: Time Series Extraction & Moving Statistics
        cpu_ts = self.history.get_time_series("cpu_utilization")
        cpu_vals = [pt[1] for pt in cpu_ts]
        cpu_mean, cpu_std, cpu_min, cpu_max = self.statistics.compute_summary_stats(cpu_vals)

        mem_ts = self.history.get_time_series("heap_utilization")
        mem_vals = [pt[1] for pt in mem_ts]
        memory_mean, memory_std, _, _ = self.statistics.compute_summary_stats(mem_vals)

        # Latency time series
        avg_inf_ts = self.history.get_time_series("average_inference_time")
        if avg_inf_ts:
            lat_ts = avg_inf_ts
        else:
            lat_ts = self.history.get_time_series("inference_time")
        lat_vals = [pt[1] for pt in lat_ts]
        latency_mean, _, _, latency_max = self.statistics.compute_summary_stats(lat_vals)

        deadline_miss_rate = self.statistics.calculate_deadline_miss_rate(self.history)

        stats_dict = {
            "cpu_mean": cpu_mean,
            "cpu_std": cpu_std,
            "cpu_min": cpu_min,
            "cpu_max": cpu_max,
            "memory_mean": memory_mean,
            "memory_std": memory_std,
            "latency_mean": latency_mean,
            "latency_max": latency_max,
            "deadline_miss_rate": deadline_miss_rate,
        }

        # Step 5: Trend Analysis (OLS Regression Slope)
        cpu_trend = self.statistics.classify_trend(
            cpu_ts,
            degrading_threshold=self.config.cpu.trend_degrading_slope,
            improving_threshold=self.config.cpu.trend_improving_slope,
            higher_is_degrading=True
        )

        memory_trend = self.statistics.classify_trend(
            mem_ts,
            degrading_threshold=self.config.memory.trend_degrading_slope,
            improving_threshold=self.config.memory.trend_improving_slope,
            higher_is_degrading=True
        )
        heap_trend = memory_trend # Heap utilization represents memory in heap-based RTOS

        stack_ts = self.history.get_time_series("stack_utilization")
        stack_trend = self.statistics.classify_trend(
            stack_ts,
            degrading_threshold=self.config.stack.trend_degrading_slope,
            improving_threshold=self.config.stack.trend_improving_slope,
            higher_is_degrading=True
        )

        latency_trend = self.statistics.classify_trend(
            lat_ts,
            degrading_threshold=self.config.ai.trend_degrading_slope,
            improving_threshold=self.config.ai.trend_improving_slope,
            higher_is_degrading=True
        )

        delay_ts = self.history.get_time_series("scheduler_delay")
        task_delay_trend = self.statistics.classify_trend(
            delay_ts,
            degrading_threshold=self.config.timing.trend_degrading_slope,
            improving_threshold=self.config.timing.trend_improving_slope,
            higher_is_degrading=True
        )

        miss_ts = self.history.get_time_series("deadline_misses")
        deadline_trend = self.statistics.classify_trend(
            miss_ts,
            degrading_threshold=0.01,
            improving_threshold=-0.01,
            higher_is_degrading=True
        )

        queue_ts = self.history.get_time_series("queue_utilization")
        queue_trend = self.statistics.classify_trend(
            queue_ts,
            degrading_threshold=self.config.queue.trend_degrading_slope,
            improving_threshold=self.config.queue.trend_improving_slope,
            higher_is_degrading=True
        )

        trends_dict = {
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "heap_trend": heap_trend,
            "stack_trend": stack_trend,
            "latency_trend": latency_trend,
            "task_delay_trend": task_delay_trend,
            "deadline_trend": deadline_trend,
            "queue_trend": queue_trend,
        }

        # Step 6: Growth Rates (% per second, positive = physical increase)
        cpu_growth_rate = self.statistics.compute_growth_rate(cpu_ts)
        memory_growth_rate = self.statistics.compute_growth_rate(mem_ts)
        heap_growth_rate = memory_growth_rate
        stack_growth_rate = self.statistics.compute_growth_rate(stack_ts)
        latency_growth_rate = self.statistics.compute_growth_rate(lat_ts)
        queue_growth_rate = self.statistics.compute_growth_rate(queue_ts)

        growth_rates_dict = {
            "cpu_growth_rate": cpu_growth_rate,
            "memory_growth_rate": memory_growth_rate,
            "heap_growth_rate": heap_growth_rate,
            "stack_growth_rate": stack_growth_rate,
            "latency_growth_rate": latency_growth_rate,
            "queue_growth_rate": queue_growth_rate,
        }

        # Step 7: Domain Health Scores (0 - 100) & Scenario-Weighted Overall Health
        health_scores = self.health.calculate_domain_scores(
            reading=cleaned_data,
            features=derived_feat,
            stats=stats_dict,
            trends=trends_dict,
            growth_rates=growth_rates_dict,
        )

        # Step 8: Fault Detection, Root-Cause Diagnosis, Severity & Fault Count
        fault_results: FaultAnalysisResult = self.faults.analyze_faults(
            reading=cleaned_data,
            features=derived_feat,
            stats=stats_dict,
            trends=trends_dict,
            growth_rates=growth_rates_dict,
            domain_scores=health_scores,
        )

        # Step 9: Degradation Analysis, Rates & Health State Classification
        overall_health = health_scores["overall_health_score"]
        # NOTE: dict.get(key, default) only substitutes the default when the key
        # is absent, not when it's present with value None. cleaned_data always
        # contains a "timestamp" key (possibly None), so the default above was
        # silently ignored. Guard explicitly instead.
        curr_ts = cleaned_data.get("timestamp")
        if curr_ts is None:
            curr_ts = 0.0

        degradation_index = self.degradation.compute_degradation_index(
            overall_health=overall_health,
            fault_count=fault_results.fault_count,
            fault_severity=fault_results.fault_severity,
            trends=trends_dict,
        )

        deg_rate, health_rate = self.degradation.update_and_calculate_rates(
            current_ts=curr_ts,
            degradation_index=degradation_index,
            overall_health=overall_health,
            min_trend_points=self.config.window.min_trend_points
        )

        health_state = self.degradation.classify_health_state(
            overall_health=overall_health,
            degradation_index=degradation_index,
            fault_severity=fault_results.fault_severity,
            fault_count=fault_results.fault_count,
            degradation_rate=deg_rate,
        )

        # Step 10: Diagnostic Tracing
        self._record_diagnostics(
            cleaned_data=cleaned_data,
            stats_dict=stats_dict,
            trends_dict=trends_dict,
            health_scores=health_scores,
            fault_results=fault_results,
            degradation_index=degradation_index,
            confidence=cleaned_reading.confidence,
        )

        # Step 11: Construct Production Output JSON matching EXACTLY the 47 fields
        output: Dict[str, Any] = {
            "timestamp": curr_ts,
            "cpu_mean": cpu_mean,
            "cpu_std": cpu_std,
            "cpu_min": cpu_min,
            "cpu_max": cpu_max,
            "cpu_trend": cpu_trend,
            "cpu_growth_rate": cpu_growth_rate,
            "memory_mean": memory_mean,
            "memory_std": memory_std,
            "memory_trend": memory_trend,
            "memory_growth_rate": memory_growth_rate,
            "heap_trend": heap_trend,
            "heap_growth_rate": heap_growth_rate,
            "stack_trend": stack_trend,
            "stack_growth_rate": stack_growth_rate,
            "latency_mean": latency_mean,
            "latency_max": latency_max,
            "latency_trend": latency_trend,
            "latency_growth_rate": latency_growth_rate,
            "task_delay_trend": task_delay_trend,
            "deadline_miss_rate": deadline_miss_rate,
            "deadline_trend": deadline_trend,
            "queue_trend": queue_trend,
            "queue_growth_rate": queue_growth_rate,
            "cpu_health_score": health_scores["cpu_health_score"],
            "memory_health_score": health_scores["memory_health_score"],
            "stack_health_score": health_scores["stack_health_score"],
            "task_health_score": health_scores["task_health_score"],
            "timing_health_score": health_scores["timing_health_score"],
            "ai_health_score": health_scores["ai_health_score"],
            "resource_health_score": health_scores["resource_health_score"],
            "overall_health_score": health_scores["overall_health_score"],
            "health_state": health_state,
            "cpu_overload_flag": fault_results.cpu_overload_flag,
            "memory_leak_flag": fault_results.memory_leak_flag,
            "heap_exhaustion_flag": fault_results.heap_exhaustion_flag,
            "stack_risk_flag": fault_results.stack_risk_flag,
            "deadline_miss_flag": fault_results.deadline_miss_flag,
            "task_starvation_flag": fault_results.task_starvation_flag,
            "ai_latency_flag": fault_results.ai_latency_flag,
            "queue_overflow_flag": fault_results.queue_overflow_flag,
            "fault_type": fault_results.fault_type,
            "fault_severity": fault_results.fault_severity,
            "fault_count": fault_results.fault_count,
            "degradation_index": degradation_index,
            "degradation_rate": deg_rate,
            "health_change_rate": health_rate,
        }

        # Step 12: JSON Sanity Enforcer (no NaN, no Inf, verify key completeness)
        clean_output = self._sanitize_for_json(output)

        if return_diagnostics:
            return clean_output, self.diagnostics.get_all_explanations()
        return clean_output

    def _build_invalid_input_output(self, cleaned_reading: CleanedReading) -> Dict[str, Any]:
        """
        Build a safe, fully contract-compliant (47-key) output for input that the
        validator could not anchor in time (no timestamp and no uptime_ms). All
        numeric/statistical fields are None, booleans are False, fault_type is
        NONE, and health_state is UNKNOWN so downstream consumers can distinguish
        this from an actual healthy reading rather than receiving a crash.
        """
        output: Dict[str, Any] = {key: None for key in PRODUCTION_OUTPUT_KEYS}
        output.update({
            "timestamp": 0.0,
            "cpu_trend": "INSUFFICIENT_DATA",
            "memory_trend": "INSUFFICIENT_DATA",
            "heap_trend": "INSUFFICIENT_DATA",
            "stack_trend": "INSUFFICIENT_DATA",
            "latency_trend": "INSUFFICIENT_DATA",
            "task_delay_trend": "INSUFFICIENT_DATA",
            "deadline_miss_rate": 0.0,
            "deadline_trend": "INSUFFICIENT_DATA",
            "queue_trend": "INSUFFICIENT_DATA",
            "health_state": "UNKNOWN",
            "cpu_overload_flag": False,
            "memory_leak_flag": False,
            "heap_exhaustion_flag": False,
            "stack_risk_flag": False,
            "deadline_miss_flag": False,
            "task_starvation_flag": False,
            "ai_latency_flag": False,
            "queue_overflow_flag": False,
            "fault_type": "NONE",
            "fault_severity": "NONE",
            "fault_count": 0,
            "degradation_index": 1.0,
        })

        self.diagnostics.record_explanation(
            MetricExplanation(
                metric="overall_health_score",
                value=None,
                formula="N/A - input rejected before scoring",
                input_metrics_used=[],
                derived_features_used=[],
                normalized_values={},
                weights_contributions={},
                historical_window_points=self.history.history_length(),
                confidence=cleaned_reading.confidence,
                reason=(
                    "Input rejected: "
                    + "; ".join(cleaned_reading.anomalies or ["MISSING_TEMPORAL_ANCHOR"])
                ),
            )
        )
        return self._sanitize_for_json(output)

    def get_diagnostics(self, metric_name: Optional[str] = None) -> Any:
        """Retrieve diagnostic traces without modifying the stream or output contract."""
        if metric_name:
            return self.diagnostics.get_explanation(metric_name)
        return self.diagnostics.get_all_explanations()

    def _sanitize_for_json(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all fields conform to strict valid JSON standards (no NaN, no Infinity)."""
        sanitized = {}
        for key in PRODUCTION_OUTPUT_KEYS:
            val = output.get(key)
            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    sanitized[key] = None
                else:
                    sanitized[key] = val
            else:
                sanitized[key] = val
        return sanitized

    def _record_diagnostics(
        self,
        cleaned_data: Dict[str, Any],
        stats_dict: Dict[str, Any],
        trends_dict: Dict[str, str],
        health_scores: Dict[str, float],
        fault_results: FaultAnalysisResult,
        degradation_index: float,
        confidence: float,
    ) -> None:
        """Populate non-intrusive diagnostic metadata."""
        hist_len = self.history.history_length()

        # Record overall health trace
        self.diagnostics.record_explanation(
            MetricExplanation(
                metric="overall_health_score",
                value=health_scores["overall_health_score"],
                formula="Scenario-weighted harmonic/bottleneck aggregation across 7 domains",
                input_metrics_used=["cpu_utilization", "heap_utilization", "stack_utilization", "scheduler_delay", "average_inference_time"],
                derived_features_used=["cpu_headroom", "heap_headroom", "ai_latency_pressure"],
                normalized_values=health_scores,
                weights_contributions=self.config.get_scenario_profile(cleaned_data.get("scenario_id")).__dict__,
                historical_window_points=hist_len,
                confidence=confidence,
                reason=f"Overall health computed with bottleneck minimum domain {min(health_scores.values()):.1f}",
            )
        )

        # Record root cause trace
        self.diagnostics.record_explanation(
            MetricExplanation(
                metric="fault_type",
                value=fault_results.fault_type,
                formula="Argmax evidence score over 11 diagnostic categories with multi-fault detection",
                input_metrics_used=["cpu_utilization", "heap_utilization", "dropped_messages", "deadline_misses"],
                derived_features_used=["queue_pressure", "dropped_message_ratio", "ai_latency_pressure"],
                normalized_values=fault_results.cause_contributions,
                weights_contributions={},
                historical_window_points=hist_len,
                confidence=confidence,
                reason=f"Dominant cause: {fault_results.fault_type} with severity {fault_results.fault_severity} (Active flags: {fault_results.fault_count})",
            )
        )
