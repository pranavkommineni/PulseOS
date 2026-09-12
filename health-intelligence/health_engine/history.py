"""
Historical State Buffer and Device Reset Detection Engine.

Maintains time-indexed historical observations over rolling windows,
detects device restarts and counter discontinuities, and manages baseline resets.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class HistoryEntry:
    """Historical record stored in the rolling buffer."""
    timestamp: float
    uptime_ms: float
    sample_id: Optional[int]
    scenario_id: Optional[Any]
    metrics: Dict[str, Any]
    confidence: float


class HistoryManager:
    """
    Manages rolling time-window history, detects device reboots and counter resets,
    and isolates statistics across reboot boundaries.
    """

    def __init__(self, config=None):
        self.config = config
        self.buffer: deque[HistoryEntry] = deque()
        self.last_uptime_ms: Optional[float] = None
        self.last_timestamp: Optional[float] = None
        self.last_sample_id: Optional[int] = None
        self.last_watchdog_resets: Optional[int] = None
        self.last_system_resets: Optional[int] = None
        self.last_scenario_id: Optional[Any] = None

        self.reset_detected: bool = False
        self.reset_reason: Optional[str] = None
        self.total_samples_processed: int = 0
        self.active_session_samples: int = 0

    def add_reading(self, cleaned_data: Dict[str, Any], confidence: float = 1.0) -> Tuple[bool, Optional[str]]:
        """
        Evaluate incoming reading for restart/discontinuity.
        Updates internal historical buffer and returns (is_reset, reset_reason).
        """
        curr_ts = cleaned_data.get("timestamp")
        curr_uptime = cleaned_data.get("uptime_ms")
        curr_sample_id = cleaned_data.get("sample_id")
        curr_wd_resets = cleaned_data.get("watchdog_resets")
        curr_sys_resets = cleaned_data.get("system_resets")
        curr_scenario = cleaned_data.get("scenario_id")

        is_reset = False
        reset_reason = None

        # Check 1: Uptime regression (device reboot)
        if self.last_uptime_ms is not None and curr_uptime is not None:
            if curr_uptime < (self.last_uptime_ms - 500.0): # Dropped by more than 500ms
                is_reset = True
                reset_reason = f"UPTIME_REGRESSION: previous={self.last_uptime_ms}ms, current={curr_uptime}ms"

        # Check 2: Explicit reset counters incremented
        if not is_reset and self.last_system_resets is not None and curr_sys_resets is not None:
            if curr_sys_resets > self.last_system_resets:
                is_reset = True
                reset_reason = f"SYSTEM_RESET_INCREMENT: {self.last_system_resets} -> {curr_sys_resets}"

        if not is_reset and self.last_watchdog_resets is not None and curr_wd_resets is not None:
            if curr_wd_resets > self.last_watchdog_resets:
                is_reset = True
                reset_reason = f"WATCHDOG_RESET_INCREMENT: {self.last_watchdog_resets} -> {curr_wd_resets}"

        # Check 3: Sample ID reset/backward jump
        if not is_reset and self.last_sample_id is not None and curr_sample_id is not None:
            if curr_sample_id < self.last_sample_id and self.last_sample_id > 5:
                is_reset = True
                reset_reason = f"SAMPLE_ID_RESET: previous={self.last_sample_id}, current={curr_sample_id}"

        # Check 4: Large timestamp discontinuity
        window_cfg = self.config.window if self.config else None
        gap_threshold = window_cfg.reset_gap_threshold_sec if window_cfg else 30.0
        if not is_reset and self.last_timestamp is not None and curr_ts is not None:
            time_delta = curr_ts - self.last_timestamp
            if time_delta < -1.0: # Negative time flow
                is_reset = True
                reset_reason = f"NEGATIVE_TIME_JUMP: {self.last_timestamp} -> {curr_ts}"
            elif time_delta > gap_threshold: # Offline gap
                is_reset = True
                reset_reason = f"OFFLINE_GAP_EXCEEDED: gap={time_delta:.1f}s > {gap_threshold}s"

        # Action on reset: flush buffer so trends are not calculated across reboots
        if is_reset:
            self.buffer.clear()
            self.active_session_samples = 0
            self.reset_detected = True
            self.reset_reason = reset_reason
        else:
            self.reset_detected = False
            self.reset_reason = None

        # Record entry
        entry = HistoryEntry(
            timestamp=curr_ts if curr_ts is not None else 0.0,
            uptime_ms=curr_uptime if curr_uptime is not None else 0.0,
            sample_id=curr_sample_id,
            scenario_id=curr_scenario,
            metrics=dict(cleaned_data),
            confidence=confidence,
        )
        self.buffer.append(entry)
        self.total_samples_processed += 1
        self.active_session_samples += 1

        # Update tracking anchors
        if curr_uptime is not None:
            self.last_uptime_ms = curr_uptime
        if curr_ts is not None:
            self.last_timestamp = curr_ts
        if curr_sample_id is not None:
            self.last_sample_id = curr_sample_id
        if curr_wd_resets is not None:
            self.last_watchdog_resets = curr_wd_resets
        if curr_sys_resets is not None:
            self.last_system_resets = curr_sys_resets
        self.last_scenario_id = curr_scenario

        # Prune old entries outside window_duration_sec or exceeding max_history_points
        self._prune_history(curr_ts)

        return is_reset, reset_reason

    def _prune_history(self, current_ts: Optional[float]) -> None:
        """Evict records older than configured time window or exceeding max count."""
        max_points = self.config.window.max_history_points if self.config else 300
        window_sec = self.config.window.window_duration_sec if self.config else 120.0

        # Prune by count
        while len(self.buffer) > max_points:
            self.buffer.popleft()

        # Prune by timestamp
        if current_ts is not None:
            cutoff = current_ts - window_sec
            while self.buffer and self.buffer[0].timestamp < cutoff:
                self.buffer.popleft()

    def get_time_series(self, metric_name: str) -> List[Tuple[float, Any]]:
        """
        Extract valid (timestamp, value) pairs for a specific metric across history.
        Omit missing / null observations without fabricating data.
        """
        series = []
        for entry in self.buffer:
            val = entry.metrics.get(metric_name)
            if val is not None:
                series.append((entry.timestamp, val))
        return series

    def get_latest_reading(self) -> Optional[HistoryEntry]:
        """Return the most recent entry, or None if buffer is empty."""
        return self.buffer[-1] if self.buffer else None

    def history_length(self) -> int:
        """Return count of active entries in window."""
        return len(self.buffer)

    def reset_state(self) -> None:
        """Manually clear all historical state (for testing or clean reboot)."""
        self.buffer.clear()
        self.last_uptime_ms = None
        self.last_timestamp = None
        self.last_sample_id = None
        self.last_watchdog_resets = None
        self.last_system_resets = None
        self.last_scenario_id = None
        self.reset_detected = False
        self.reset_reason = None
        self.active_session_samples = 0
