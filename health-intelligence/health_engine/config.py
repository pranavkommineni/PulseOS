"""
Configuration and Calibration Parameter System for Person 2 Health Intelligence Engine.

Distinguishes between FIXED PHYSICAL/LOGICAL RELATIONSHIPS and CALIBRATABLE PARAMETERS.
Allows full customization, runtime calibration, and scenario profiling.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class WindowConfig:
    """Historical window configuration parameters."""
    window_duration_sec: float = 120.0       # Standard analysis time window (seconds)
    min_history_points: int = 3              # Minimum observations required for running statistics
    min_trend_points: int = 5                # Minimum observations required for OLS trend calculation
    max_history_points: int = 300            # Maximum historical buffer depth (records)
    reset_gap_threshold_sec: float = 30.0    # Time gap indicating potential offline/discontinuity period


@dataclass
class CPUConfig:
    """CPU calibration thresholds and envelope boundaries."""
    healthy_upper_pct: float = 70.0          # Upper bound of normal CPU operating range
    overload_threshold_pct: float = 88.0     # Sustained CPU utilization threshold for CPU_OVERLOAD
    spike_headroom_pct: float = 95.0         # Extreme saturation boundary
    trend_degrading_slope: float = 0.10      # Slope (%/sec) above which CPU is classified as DEGRADING
    trend_improving_slope: float = -0.10     # Slope (%/sec) below which CPU is classified as IMPROVING
    growth_rate_warning: float = 0.5         # Growth rate (%/sec) warning boundary


@dataclass
class MemoryConfig:
    """Memory and Heap calibration thresholds."""
    heap_healthy_upper_pct: float = 75.0     # Upper bound of healthy heap utilization
    heap_warning_pct: float = 85.0           # Warning threshold for heap consumption
    heap_exhaustion_pct: float = 92.0        # Hard threshold for HEAP_EXHAUSTION flag
    critical_min_free_heap_bytes: int = 8192 # Critical floor for minimum_free_heap (8 KB)
    leak_min_slope_pct_per_sec: float = 0.02 # Positive slope over time indicating sustained leak
    leak_min_duration_sec: float = 20.0      # Duration over which heap must grow monotonically for leak
    trend_degrading_slope: float = 0.05      # Degrading slope threshold for heap (%/sec)
    trend_improving_slope: float = -0.05


@dataclass
class StackConfig:
    """Stack watermark and utilization calibration."""
    stack_healthy_upper_pct: float = 75.0    # Normal stack utilization upper limit
    stack_risk_pct: float = 88.0             # STACK_RISK trigger threshold
    critical_watermark_bytes: int = 256      # Critical minimum free stack margin (bytes/words)
    trend_degrading_slope: float = 0.05
    trend_improving_slope: float = -0.05


@dataclass
class TimingTaskConfig:
    """RTOS task and timing calibration."""
    max_acceptable_scheduler_delay_ms: float = 15.0 # Delay threshold indicating scheduling congestion
    max_acceptable_jitter_ms: float = 10.0          # Acceptable task jitter limit
    deadline_miss_rate_warning: float = 0.005       # 0.5% deadline miss rate threshold
    deadline_miss_rate_critical: float = 0.02       # 2.0% deadline miss rate threshold
    task_starvation_min_delay_ms: float = 30.0      # Starvation indicator threshold
    trend_degrading_slope: float = 0.05
    trend_improving_slope: float = -0.05


@dataclass
class AIInferenceConfig:
    """Edge-AI Inference latency and workload calibration."""
    nominal_inference_time_ms: float = 35.0  # Nominal baseline inference latency
    latency_warning_factor: float = 1.35     # Latency ratio (observed / nominal) triggering warning
    latency_critical_factor: float = 1.80    # Latency ratio triggering AI_LATENCY flag
    trend_degrading_slope: float = 0.10      # Latency degradation slope (ms/sec)
    trend_improving_slope: float = -0.10


@dataclass
class QueueCommunicationConfig:
    """RTOS message queue and inter-task communication calibration."""
    queue_healthy_upper_pct: float = 70.0    # Normal queue fullness
    queue_warning_pct: float = 85.0          # Queue pressure threshold
    queue_overflow_pct: float = 95.0         # QUEUE_OVERFLOW flag threshold
    max_dropped_message_rate: float = 0.01   # Tolerable dropped message rate
    trend_degrading_slope: float = 0.10
    trend_improving_slope: float = -0.10


@dataclass
class HardwareResourceConfig:
    """Thermal, power, and interrupt hardware calibration."""
    temp_nominal_celsius: float = 45.0       # Typical baseline SoC operating temperature
    temp_warning_celsius: float = 72.0       # Thermal throttling warning
    temp_critical_celsius: float = 85.0      # Thermal danger threshold
    power_nominal_mw: float = 1500.0         # Nominal power draw
    power_critical_mw: float = 3500.0        # Peak power limit
    interrupt_latency_critical_ms: float = 5.0 # Interrupt handling delay alert


@dataclass
class ScenarioProfile:
    """Operating profile modifying domain weights and expectations per scenario_id."""
    name: str
    weight_cpu: float = 0.18
    weight_memory: float = 0.18
    weight_stack: float = 0.10
    weight_task: float = 0.14
    weight_timing: float = 0.14
    weight_ai: float = 0.14
    weight_resource: float = 0.12


@dataclass
class HealthStateThresholds:
    """Thresholds mapping overall_health_score and degradation_index to health_state."""
    healthy_min_score: float = 85.0
    healthy_max_degradation: float = 0.18
    stressed_min_score: float = 68.0
    stressed_max_degradation: float = 0.35
    degrading_min_score: float = 48.0
    degrading_max_degradation: float = 0.60
    critical_min_score: float = 25.0
    critical_max_degradation: float = 0.82
    # Below critical_min_score or above critical_max_degradation -> FAILURE_RISK


class health_engineConfig:
    """
    Master Configuration class aggregating all modular configurations.
    Provides methods to export, load, and calibrate parameters at runtime.
    """

    def __init__(
        self,
        window: Optional[WindowConfig] = None,
        cpu: Optional[CPUConfig] = None,
        memory: Optional[MemoryConfig] = None,
        stack: Optional[StackConfig] = None,
        timing: Optional[TimingTaskConfig] = None,
        ai: Optional[AIInferenceConfig] = None,
        queue: Optional[QueueCommunicationConfig] = None,
        hardware: Optional[HardwareResourceConfig] = None,
        health_states: Optional[HealthStateThresholds] = None,
        scenario_profiles: Optional[Dict[str, ScenarioProfile]] = None,
    ):
        self.window = window or WindowConfig()
        self.cpu = cpu or CPUConfig()
        self.memory = memory or MemoryConfig()
        self.stack = stack or StackConfig()
        self.timing = timing or TimingTaskConfig()
        self.ai = ai or AIInferenceConfig()
        self.queue = queue or QueueCommunicationConfig()
        self.hardware = hardware or HardwareResourceConfig()
        self.health_states = health_states or HealthStateThresholds()

        # Configurable scenario profiles mapping scenario_id -> profile
        self.scenario_profiles: Dict[str, ScenarioProfile] = scenario_profiles or {
            "default": ScenarioProfile(
                name="Default Balanced",
                weight_cpu=0.18,
                weight_memory=0.18,
                weight_stack=0.10,
                weight_task=0.14,
                weight_timing=0.14,
                weight_ai=0.14,
                weight_resource=0.12,
            ),
            "1": ScenarioProfile(
                name="Normal Sensing & Telemetry",
                weight_cpu=0.20,
                weight_memory=0.20,
                weight_stack=0.10,
                weight_task=0.15,
                weight_timing=0.15,
                weight_ai=0.10,
                weight_resource=0.10,
            ),
            "2": ScenarioProfile(
                name="AI-Heavy Vision / Model Execution",
                weight_cpu=0.25,
                weight_memory=0.15,
                weight_stack=0.08,
                weight_task=0.10,
                weight_timing=0.12,
                weight_ai=0.25,
                weight_resource=0.05,
            ),
            "3": ScenarioProfile(
                name="Communication / Burst Queue Workload",
                weight_cpu=0.15,
                weight_memory=0.20,
                weight_stack=0.08,
                weight_task=0.15,
                weight_timing=0.12,
                weight_ai=0.08,
                weight_resource=0.22, # Queue and bus congestion pressure
            ),
            "4": ScenarioProfile(
                name="Low-Power Standby / Edge Sensing",
                weight_cpu=0.15,
                weight_memory=0.15,
                weight_stack=0.10,
                weight_task=0.10,
                weight_timing=0.10,
                weight_ai=0.05,
                weight_resource=0.35, # Power consumption and thermal constraints
            ),
        }

    def get_scenario_profile(self, scenario_id: Any) -> ScenarioProfile:
        """Retrieve the configured profile for a scenario_id, or return default."""
        sid_str = str(scenario_id) if scenario_id is not None else "default"
        return self.scenario_profiles.get(sid_str, self.scenario_profiles["default"])

    def register_scenario(self, scenario_id: str, profile: ScenarioProfile) -> None:
        """Register or update a custom scenario profile."""
        self.scenario_profiles[str(scenario_id)] = profile

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Runtime calibration: update parameters from nested dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section) and isinstance(values, dict):
                target_obj = getattr(self, section)
                for k, v in values.items():
                    if hasattr(target_obj, k):
                        setattr(target_obj, k, v)
