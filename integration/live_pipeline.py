import argparse
import csv
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

INTEGRATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INTEGRATION_DIR.parent

HEALTH_INTEL_DIR = PROJECT_ROOT / "health-intelligence"
RSUL_CONTROL_SRC_DIR = PROJECT_ROOT / "rsul-control" / "src"

if str(HEALTH_INTEL_DIR) not in sys.path:
    sys.path.insert(0, str(HEALTH_INTEL_DIR))

if str(RSUL_CONTROL_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(RSUL_CONTROL_SRC_DIR))


# ============================================================================
# IMPORT SERIAL
# ============================================================================

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


# ============================================================================
# IMPORT HEALTH INTELLIGENCE
# ============================================================================

try:
    from health_engine.engine import (
        HealthIntelligenceEngine,
        PRODUCTION_OUTPUT_KEYS,
    )
except ImportError as e:
    sys.exit(
        "Could not import HealthIntelligenceEngine from health-intelligence/. "
        f"Run this script from the PulseOS-main repo root. ({e})"
    )


# ============================================================================
# IMPORT RSUL
# ============================================================================

try:
    from rsul_control.prediction.engine import predict_payload
except ImportError as e:
    sys.exit(
        "Could not import predict_payload from rsul-control/. "
        f"Run this script from the PulseOS-main repo root, and "
        f"`pip install -r rsul-control/requirements.txt`. ({e})"
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

BAUD_RATE = 115200

RAW_CSV_PATH = "amr_dataset.csv"

OUTPUT_CSV_PATH = "health_intelligence_output.csv"

RSUL_OUTPUT_CSV_PATH = "rsul_control_output.csv"

# Rolling history used to calculate trends.
TREND_WINDOW = 10

# Minimum history before slope calculations become meaningful.
MIN_TREND_SAMPLES = 3

HEARTBEAT_SECONDS = 5.0


# ============================================================================
# RSUL OUTPUT SCHEMA
# ============================================================================

RSUL_OUTPUT_KEYS = [
    "timestamp",
    "current_health",
    "health_state",
    "current_degradation_rate",
    "predicted_health",
    "predicted_degradation_rate",
    "predicted_critical_time",
    "failure_probability",
    "risk_level",
    "predicted_rsul_hours",
    "predicted_failure_time",
    "rsul_confidence",
    "dominant_degradation_factor",
    "factor_contribution",
    "recommendation",
    "recommendation_priority",
    "compatibility_mode",
]


# ============================================================================
# ESP32-S3 INPUT SCHEMA
# ============================================================================

EXPECTED_HEADER_PREFIX = "timestamp,sample_id,uptime_ms,scenario_id"


HEADER_COLS = [
    # ------------------------------------------------------------------------
    # Core telemetry
    # ------------------------------------------------------------------------
    "timestamp",
    "sample_id",
    "uptime_ms",
    "scenario_id",
    # ------------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------------
    "cpu_utilization",
    "cpu_idle",
    "task_cpu_utilization",
    # ------------------------------------------------------------------------
    # Memory / heap
    # ------------------------------------------------------------------------
    "total_heap",
    "free_heap",
    "used_heap",
    "heap_utilization",
    "minimum_free_heap",
    # ------------------------------------------------------------------------
    # Stack
    # ------------------------------------------------------------------------
    "stack_high_water_mark",
    "stack_utilization",
    # ------------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------------
    "task_name",
    "task_priority",
    "task_state",
    "task_execution_time",
    "task_period",
    "task_jitter",
    "task_execution_count",
    "deadline_misses",
    # ------------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------------
    "context_switches",
    "task_switches",
    "scheduler_delay",
    "active_task_count",
    # ------------------------------------------------------------------------
    # AI inference
    # ------------------------------------------------------------------------
    "inference_time",
    "min_inference_time",
    "max_inference_time",
    "average_inference_time",
    "inference_count",
    "inference_frequency",
    # ------------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------------
    "queue_length",
    "queue_capacity",
    "queue_utilization",
    # ------------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------------
    "messages_sent",
    "messages_received",
    "dropped_messages",
    # ------------------------------------------------------------------------
    # Interrupts
    # ------------------------------------------------------------------------
    "interrupt_count",
    "interrupt_latency",
    # ------------------------------------------------------------------------
    # Power / temperature
    # ------------------------------------------------------------------------
    "power_consumption",
    "system_temperature",
    # ------------------------------------------------------------------------
    # Reset counters
    # ------------------------------------------------------------------------
    "watchdog_resets",
    "system_resets",
    # ------------------------------------------------------------------------
    # Additional firmware metrics
    # ------------------------------------------------------------------------
    "heap_churn_bytes",
    "heap_alloc_failures",
    "distance_cm",
    "ultrasonic_timeouts",
]


# ============================================================================
# DERIVED PERSON-2 METRICS
# ============================================================================

DERIVED_KEYS = [
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
    "fault_severity",
]


# ============================================================================
# TREND HISTORY
# ============================================================================


class TrendTracker:
    """
    Maintains a rolling window of telemetry and calculates:

        mean
        std
        min
        max
        slope/trend
        normalized growth rate
    """

    def __init__(self, window_size=TREND_WINDOW):

        self.window_size = window_size

        self.history = {
            "cpu": deque(maxlen=window_size),
            "memory": deque(maxlen=window_size),
            "heap": deque(maxlen=window_size),
            "stack": deque(maxlen=window_size),
            "latency": deque(maxlen=window_size),
            "task_delay": deque(maxlen=window_size),
            "deadline": deque(maxlen=window_size),
            "queue": deque(maxlen=window_size),
        }

    # ------------------------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------------------------

    @staticmethod
    def to_float(value, default=0.0):

        try:
            value = float(value)

            if not math.isfinite(value):
                return default

            return value

        except (TypeError, ValueError):

            return default

    # ------------------------------------------------------------------------
    # Add sample
    # ------------------------------------------------------------------------

    def add(self, reading):

        cpu = self.to_float(reading.get("cpu_utilization"))

        memory = self.to_float(reading.get("heap_utilization"))

        heap = self.to_float(reading.get("heap_utilization"))

        stack = self.to_float(reading.get("stack_utilization"))

        latency = self.to_float(reading.get("inference_time"))

        task_delay = self.to_float(reading.get("scheduler_delay"))

        deadline = self.to_float(reading.get("deadline_misses"))

        queue = self.to_float(reading.get("queue_utilization"))

        self.history["cpu"].append(cpu)
        self.history["memory"].append(memory)
        self.history["heap"].append(heap)
        self.history["stack"].append(stack)
        self.history["latency"].append(latency)
        self.history["task_delay"].append(task_delay)
        self.history["deadline"].append(deadline)
        self.history["queue"].append(queue)

    # ------------------------------------------------------------------------
    # Linear slope
    # ------------------------------------------------------------------------

    @staticmethod
    def slope(values):

        if len(values) < MIN_TREND_SAMPLES:
            return 0.0

        y = list(values)

        n = len(y)

        x_mean = (n - 1) / 2.0
        y_mean = sum(y) / n

        numerator = 0.0
        denominator = 0.0

        for i, value in enumerate(y):

            dx = i - x_mean
            dy = value - y_mean

            numerator += dx * dy
            denominator += dx * dx

        if denominator == 0:
            return 0.0

        result = numerator / denominator

        if not math.isfinite(result):
            return 0.0

        return result

    # ------------------------------------------------------------------------
    # Growth rate
    # ------------------------------------------------------------------------

    @classmethod
    def growth_rate(cls, values):

        if len(values) < MIN_TREND_SAMPLES:
            return 0.0

        values = list(values)

        baseline = abs(values[0])

        slope = cls.slope(values)

        # Prevent division by zero.
        denominator = max(baseline, 1e-6)

        rate = slope / denominator

        if not math.isfinite(rate):
            return 0.0

        return rate

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    @staticmethod
    def statistics(values):

        if not values:

            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        values = list(values)

        mean = sum(values) / len(values)

        if len(values) > 1:

            variance = sum((x - mean) ** 2 for x in values) / len(values)

            std = math.sqrt(max(variance, 0.0))

        else:

            std = 0.0

        return {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
        }

    # ------------------------------------------------------------------------
    # Get derived metrics
    # ------------------------------------------------------------------------

    def metrics(self):

        cpu_stats = self.statistics(self.history["cpu"])
        memory_stats = self.statistics(self.history["memory"])
        latency_stats = self.statistics(self.history["latency"])

        return {
            # CPU
            "cpu_mean": cpu_stats["mean"],
            "cpu_std": cpu_stats["std"],
            "cpu_min": cpu_stats["min"],
            "cpu_max": cpu_stats["max"],
            "cpu_trend": self.slope(self.history["cpu"]),
            "cpu_growth_rate": self.growth_rate(self.history["cpu"]),
            # Memory
            "memory_mean": memory_stats["mean"],
            "memory_std": memory_stats["std"],
            "memory_trend": self.slope(self.history["memory"]),
            "memory_growth_rate": self.growth_rate(self.history["memory"]),
            # Heap
            "heap_trend": self.slope(self.history["heap"]),
            "heap_growth_rate": self.growth_rate(self.history["heap"]),
            # Stack
            "stack_trend": self.slope(self.history["stack"]),
            "stack_growth_rate": self.growth_rate(self.history["stack"]),
            # Latency
            "latency_mean": latency_stats["mean"],
            "latency_max": latency_stats["max"],
            "latency_trend": self.slope(self.history["latency"]),
            "latency_growth_rate": self.growth_rate(self.history["latency"]),
            # Task delay
            "task_delay_trend": self.slope(self.history["task_delay"]),
            # Deadline
            "deadline_miss_rate": (
                self.history["deadline"][-1] if self.history["deadline"] else 0.0
            ),
            "deadline_trend": self.slope(self.history["deadline"]),
            # Queue
            "queue_trend": self.slope(self.history["queue"]),
            "queue_growth_rate": self.growth_rate(self.history["queue"]),
        }


# ============================================================================
# FAULT SEVERITY
# ============================================================================


def calculate_fault_severity(reading, metrics):
    """
    Calculate a continuous fault severity score from 0-1.

    0.0 = healthy
    1.0 = severe degradation

    This is intentionally derived from the same telemetry signals
    available to the live pipeline.
    """

    scores = []

    # ------------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------------

    cpu = float(metrics.get("cpu_mean", 0.0))

    cpu_score = max(0.0, min(1.0, (cpu - 70.0) / 30.0))

    scores.append(cpu_score)

    # ------------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------------

    memory = float(metrics.get("memory_mean", 0.0))

    memory_score = max(0.0, min(1.0, (memory - 70.0) / 30.0))

    scores.append(memory_score)

    # ------------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------------

    latency = float(metrics.get("latency_mean", 0.0))

    latency_score = max(0.0, min(1.0, (latency - 50.0) / 50.0))

    scores.append(latency_score)

    # ------------------------------------------------------------------------
    # Stack
    # ------------------------------------------------------------------------

    stack = TrendTracker.to_float(reading.get("stack_utilization"))

    stack_score = max(0.0, min(1.0, (stack - 70.0) / 30.0))

    scores.append(stack_score)

    # ------------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------------

    queue = TrendTracker.to_float(reading.get("queue_utilization"))

    queue_score = max(0.0, min(1.0, (queue - 70.0) / 30.0))

    scores.append(queue_score)

    # ------------------------------------------------------------------------
    # Deadline
    # ------------------------------------------------------------------------

    deadline_misses = TrendTracker.to_float(reading.get("deadline_misses"))

    deadline_score = min(1.0, deadline_misses / 5.0)

    scores.append(deadline_score)

    # ------------------------------------------------------------------------
    # Dropped messages
    # ------------------------------------------------------------------------

    dropped = TrendTracker.to_float(reading.get("dropped_messages"))

    dropped_score = min(1.0, dropped / 5.0)

    scores.append(dropped_score)

    # ------------------------------------------------------------------------
    # Growth/degradation signals
    # ------------------------------------------------------------------------

    growth_signals = [
        abs(float(metrics.get("cpu_growth_rate", 0.0))),
        abs(float(metrics.get("memory_growth_rate", 0.0))),
        abs(float(metrics.get("latency_growth_rate", 0.0))),
    ]

    for value in growth_signals:

        scores.append(min(1.0, value))

    if not scores:
        return 0.0

    severity = sum(scores) / len(scores)

    return max(0.0, min(1.0, severity))


# ============================================================================
# SCENARIO / CONDITION LABEL
# ============================================================================


def derive_condition_label(reading, metrics):
    """
    Generate a human-readable condition label.

    This is NOT used as an ML feature.

    It is useful for live monitoring and for identifying the
    condition under which telemetry was generated.
    """

    scenario = str(reading.get("scenario_id", "")).strip()

    cpu = metrics.get("cpu_mean", 0.0)
    memory = metrics.get("memory_mean", 0.0)
    latency = metrics.get("latency_mean", 0.0)

    deadline = TrendTracker.to_float(reading.get("deadline_misses"))

    queue = TrendTracker.to_float(reading.get("queue_utilization"))

    if cpu >= 90.0:
        return "CPU_OVERLOAD"

    if memory >= 90.0:
        return "MEMORY_STRESS"

    if deadline > 0:
        return "DEADLINE_STRESS"

    if queue >= 90.0:
        return "QUEUE_OVERFLOW"

    if latency >= 100.0:
        return "AI_LATENCY"

    if scenario == "4":
        return "LOW_LOAD"

    if scenario == "2":
        return "HIGH_LOAD"

    if scenario == "3":
        return "STRESS"

    if scenario == "1":
        return "NORMAL"

    return "NORMAL"


# ============================================================================
# PORT DETECTION
# ============================================================================


def find_esp32_port():

    if list_ports is None:
        return None

    ports = list(list_ports.comports())

    if not ports:
        return None

    keywords = [
        "CP210",
        "CH340",
        "CH9102",
        "FTDI",
        "USB-SERIAL",
        "USB2.0-Serial",
        "Silicon Labs",
        "USB JTAG/serial debug unit",
        "303A",
    ]

    for port in ports:

        desc = (f"{port.description} " f"{port.manufacturer or ''}").upper()

        if any(keyword.upper() in desc for keyword in keywords):

            return port.device

    if len(ports) == 1:
        return ports[0].device

    return None


# ============================================================================
# SERIAL CONNECTION
# ============================================================================


def try_connect(port):

    try:

        ser = serial.Serial()

        ser.port = port
        ser.baudrate = BAUD_RATE
        ser.timeout = 2

        ser.dtr = False
        ser.rts = False

        ser.open()

        # Reset ESP32-S3.
        ser.rts = True

        time.sleep(0.1)

        ser.rts = False
        ser.dtr = False

        time.sleep(2)

        ser.reset_input_buffer()

        return ser

    except Exception as e:

        print(f"Could not open {port}: {e}")

        return None


# ============================================================================
# SERIAL CSV -> DICT
# ============================================================================


def row_to_reading(row_line, header_cols):

    values = row_line.strip().split(",")

    if len(values) != len(header_cols):
        return None

    return dict(zip(header_cols, values))


# ============================================================================
# RAW CSV
# ============================================================================


def ensure_raw_csv_header():

    if not os.path.exists(RAW_CSV_PATH):

        with open(RAW_CSV_PATH, "w", newline="") as f:

            csv.writer(f).writerow(HEADER_COLS + ["logged_at"])


def append_raw_csv_row(reading):

    with open(RAW_CSV_PATH, "a", newline="") as f:

        csv.writer(f).writerow(
            [reading.get(key, "") for key in HEADER_COLS]
            + [reading.get("logged_at", "")]
        )


# ============================================================================
# ENRICHED HEALTH CSV HEADER
# ============================================================================


def get_enriched_output_keys():

    keys = list(PRODUCTION_OUTPUT_KEYS)

    for key in DERIVED_KEYS:

        if key not in keys:
            keys.append(key)

    extra_keys = [
        "condition_label",
        "fault_severity",
    ]

    for key in extra_keys:

        if key not in keys:
            keys.append(key)

    return keys


ENRICHED_OUTPUT_KEYS = get_enriched_output_keys()


def ensure_output_csv_header():

    if not os.path.exists(OUTPUT_CSV_PATH):

        with open(OUTPUT_CSV_PATH, "w", newline="") as f:

            csv.writer(f).writerow(ENRICHED_OUTPUT_KEYS)


def append_output_csv_row(health_json, derived_metrics):

    row = dict(health_json)

    row.update(derived_metrics)

    with open(OUTPUT_CSV_PATH, "a", newline="") as f:

        csv.writer(f).writerow([row.get(key, "") for key in ENRICHED_OUTPUT_KEYS])


# ============================================================================
# RSUL CSV
# ============================================================================


def ensure_rsul_csv_header():

    if not os.path.exists(RSUL_OUTPUT_CSV_PATH):

        with open(RSUL_OUTPUT_CSV_PATH, "w", newline="") as f:

            csv.writer(f).writerow(RSUL_OUTPUT_KEYS)


def append_rsul_csv_row(rsul_json):

    with open(RSUL_OUTPUT_CSV_PATH, "a", newline="") as f:

        csv.writer(f).writerow([rsul_json.get(key, "") for key in RSUL_OUTPUT_KEYS])


# ============================================================================
# MAIN READING PROCESSOR
# ============================================================================


def process_reading(
    engine,
    tracker,
    reading,
    convert_timestamp_ms=True,
    rsul_model="linear_regression",
    rsul_threshold=100.0,
):

    # ------------------------------------------------------------------------
    # 1. Raw logging
    # ------------------------------------------------------------------------

    logged = dict(reading)

    logged["logged_at"] = datetime.now().isoformat(timespec="seconds")

    ensure_raw_csv_header()

    append_raw_csv_row(logged)

    # ------------------------------------------------------------------------
    # 2. Update trend history
    # ------------------------------------------------------------------------

    tracker.add(reading)

    derived_metrics = tracker.metrics()

    # ------------------------------------------------------------------------
    # 3. Calculate fault severity
    # ------------------------------------------------------------------------

    fault_severity = calculate_fault_severity(reading, derived_metrics)

    derived_metrics["fault_severity"] = fault_severity

    # ------------------------------------------------------------------------
    # 4. Derive condition label
    # ------------------------------------------------------------------------

    condition_label = derive_condition_label(reading, derived_metrics)

    derived_metrics["condition_label"] = condition_label

    # ------------------------------------------------------------------------
    # 5. Build health-engine payload
    # ------------------------------------------------------------------------

    payload = dict(reading)

    if convert_timestamp_ms:

        try:

            payload["timestamp"] = float(payload["timestamp"]) / 1000.0

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            pass

    # ------------------------------------------------------------------------
    # 6. Health Intelligence
    # ------------------------------------------------------------------------

    health_json = engine.process_reading(json.dumps(payload))

    # ------------------------------------------------------------------------
    # 7. Add derived metrics to health output
    # ------------------------------------------------------------------------

    enriched_health_json = dict(health_json)

    enriched_health_json.update(derived_metrics)

    # Keep condition label explicitly available.
    enriched_health_json["condition_label"] = condition_label

    enriched_health_json["fault_severity"] = fault_severity

    # ------------------------------------------------------------------------
    # 8. Save health output
    # ------------------------------------------------------------------------

    ensure_output_csv_header()

    append_output_csv_row(health_json, derived_metrics)

    # ------------------------------------------------------------------------
    # 9. RSUL payload
    # ------------------------------------------------------------------------

    rsul_payload = dict(health_json)

    # Add derived metrics too.
    rsul_payload.update(derived_metrics)

    # rsul-control requires a real datetime.
    rsul_payload["timestamp"] = datetime.now().isoformat()

    # ------------------------------------------------------------------------
    # 10. RSUL prediction
    # ------------------------------------------------------------------------

    try:

        rsul_json = predict_payload(rsul_payload, rsul_model, rsul_threshold)

    except Exception as exc:

        print(f"WARNING: RSUL prediction failed: {exc}")

        rsul_json = {
            "timestamp": datetime.now().isoformat(),
            "current_health": health_json.get("overall_health_score", ""),
            "health_state": health_json.get("health_state", ""),
            "current_degradation_rate": health_json.get("degradation_rate", ""),
            "predicted_health": "",
            "predicted_degradation_rate": "",
            "predicted_critical_time": "",
            "failure_probability": "",
            "risk_level": "UNKNOWN",
            "predicted_rsul_hours": "",
            "predicted_failure_time": "",
            "rsul_confidence": "",
            "dominant_degradation_factor": "",
            "factor_contribution": "",
            "recommendation": "RSUL prediction unavailable",
            "recommendation_priority": "HIGH",
            "compatibility_mode": "ERROR",
        }

    # ------------------------------------------------------------------------
    # 11. Save RSUL output
    # ------------------------------------------------------------------------

    ensure_rsul_csv_header()

    append_rsul_csv_row(rsul_json)

    return (enriched_health_json, rsul_json, derived_metrics)


# ============================================================================
# TERMINAL OUTPUT
# ============================================================================


def print_health_json(health_json, rsul_json=None, derived_metrics=None):

    print(json.dumps(health_json, default=str))

    if derived_metrics:

        print("DERIVED_METRICS " + json.dumps(derived_metrics, default=str))

    if rsul_json is not None:

        print(json.dumps(rsul_json, default=str))


# ============================================================================
# LIVE MODE
# ============================================================================


def run_live(args):

    if serial is None:

        sys.exit("Missing dependency. " "Run: pip install pyserial")

    port = args.port or find_esp32_port()

    if not port:

        print("NOT LIVE: no ESP32-S3 detected " "on any USB serial port.")

        print("Plug in the ESP32-S3 or use " "--replay <csv>.")

        return

    ser = try_connect(port)

    if not ser or not ser.is_open:

        print(f"NOT LIVE: found port {port} " "but could not open a live connection.")

        return

    print(f"LIVE: connected to ESP32-S3 " f"on {port} @ {BAUD_RATE} baud")

    print("Pipeline:")

    print("ESP32-S3 -> RAW -> TRENDS -> " "HEALTH -> RSUL -> CSV")

    print(f"Trend window: {TREND_WINDOW} samples")

    print(f"Terminal interval: " f"{args.interval:g}s")

    print("Press Ctrl+C to stop.")

    # ------------------------------------------------------------------------
    # Optional ESP32 load command
    # ------------------------------------------------------------------------

    if args.load:

        cmd = {
            "low": b"L\n",
            "normal": b"N\n",
            "high": b"H\n",
        }[args.load]

        ser.write(cmd)

        print(f"Sent load_level=" f"{args.load.upper()} " "command to ESP32-S3.")

    # ------------------------------------------------------------------------
    # Initialize engines
    # ------------------------------------------------------------------------

    engine = HealthIntelligenceEngine()

    tracker = TrendTracker()

    row_count = 0
    malformed_count = 0

    last_print_ts = 0.0
    last_data_ts = time.monotonic()
    started_at = time.monotonic()

    try:

        while True:

            raw = ser.readline().decode(errors="ignore").strip()

            # ----------------------------------------------------------------
            # No data
            # ----------------------------------------------------------------

            if not raw:

                if ser.in_waiting == 0 and not ser.is_open:

                    print("NOT LIVE: serial " "connection lost.")

                    break

                if time.monotonic() - last_data_ts >= HEARTBEAT_SECONDS:

                    print(
                        f"... still waiting "
                        f"for data on {port} "
                        f"({row_count} readings)"
                    )

                    last_data_ts = time.monotonic()

                continue

            # ----------------------------------------------------------------
            # Header
            # ----------------------------------------------------------------

            if raw.startswith(EXPECTED_HEADER_PREFIX):

                continue

            # ----------------------------------------------------------------
            # Parse row
            # ----------------------------------------------------------------

            reading = row_to_reading(raw, HEADER_COLS)

            if reading is None:

                malformed_count += 1

                if malformed_count <= 3:

                    print("WARNING: malformed row, " "skipping:", raw)

                continue

            # ----------------------------------------------------------------
            # Process
            # ----------------------------------------------------------------

            (
                health_json,
                rsul_json,
                derived_metrics,
            ) = process_reading(
                engine,
                tracker,
                reading,
                convert_timestamp_ms=True,
                rsul_model=args.rsul_model,
                rsul_threshold=args.rsul_threshold,
            )

            row_count += 1

            last_data_ts = time.monotonic()

            # ----------------------------------------------------------------
            # Terminal update
            # ----------------------------------------------------------------

            now = time.monotonic()

            if now - last_print_ts >= args.interval:

                print_health_json(
                    health_json,
                    rsul_json,
                    derived_metrics,
                )

                last_print_ts = now

    except KeyboardInterrupt:

        print("\nStopping pipeline...")

    finally:

        if ser and ser.is_open:
            ser.close()

        if row_count == 0 and time.monotonic() - started_at < HEARTBEAT_SECONDS:

            print(
                "No readings were received. " "Try resetting/replugging the ESP32-S3."
            )

        print(f"{row_count} readings processed.")

        print("Final CSVs:")

        print(f"  {RAW_CSV_PATH}")

        print(f"  {OUTPUT_CSV_PATH}")

        print(f"  {RSUL_OUTPUT_CSV_PATH}")


# ============================================================================
# REPLAY MODE
# ============================================================================


def run_replay(args):

    src_path = args.replay

    if not os.path.exists(src_path):

        sys.exit(f"Replay file not found: " f"{src_path}")

    print(f"REPLAY: {src_path}")

    print("CSV -> JSON -> TRENDS -> " "HEALTH -> RSUL -> CSV")

    engine = HealthIntelligenceEngine()

    tracker = TrendTracker()

    row_count = 0

    last_print_ts = 0.0

    try:

        with open(src_path, newline="") as f:

            reader = csv.DictReader(f)

            for row in reader:

                reading = {
                    key: value for key, value in row.items() if key != "logged_at"
                }

                (
                    health_json,
                    rsul_json,
                    derived_metrics,
                ) = process_reading(
                    engine,
                    tracker,
                    reading,
                    convert_timestamp_ms=True,
                    rsul_model=args.rsul_model,
                    rsul_threshold=args.rsul_threshold,
                )

                row_count += 1

                now = time.monotonic()

                if now - last_print_ts >= args.interval:

                    print_health_json(
                        health_json,
                        rsul_json,
                        derived_metrics,
                    )

                    last_print_ts = now

                if args.replay_speed > 0:

                    time.sleep(args.replay_speed)

    except KeyboardInterrupt:

        print("\nStopping replay...")

    print(f"{row_count} readings processed.")

    print(
        f"Final CSVs: "
        f"{RAW_CSV_PATH}, "
        f"{OUTPUT_CSV_PATH}, "
        f"{RSUL_OUTPUT_CSV_PATH}"
    )


# ============================================================================
# DEMO MODE
# ============================================================================


def run_demo(args):

    try:

        from health_engine.synthetic import SyntheticTelemetryGenerator

    except ImportError as e:

        sys.exit("Could not import " f"SyntheticTelemetryGenerator. " f"({e})")

    print("DEMO:")

    print(
        "synthetic telemetry -> "
        "JSON -> TRENDS -> "
        "health-intelligence -> "
        "rsul-control -> CSV"
    )

    engine = HealthIntelligenceEngine()

    tracker = TrendTracker()

    synth = SyntheticTelemetryGenerator(seed=42)

    scenarios = [
        ("HEALTHY", 20, 0.0),
        ("CPU_OVERLOAD", 20, 20.0),
    ]

    readings = []

    for name, n, start in scenarios:

        readings.extend(
            synth.generate_scenario(
                name,
                num_samples=n,
                start_time=start,
            )
        )

    row_count = 0

    last_print_ts = 0.0

    try:

        for reading in readings:

            (
                health_json,
                rsul_json,
                derived_metrics,
            ) = process_reading(
                engine,
                tracker,
                reading,
                convert_timestamp_ms=False,
                rsul_model=args.rsul_model,
                rsul_threshold=args.rsul_threshold,
            )

            row_count += 1

            now = time.monotonic()

            if now - last_print_ts >= args.interval:

                print_health_json(
                    health_json,
                    rsul_json,
                    derived_metrics,
                )

                last_print_ts = now

            if args.replay_speed > 0:

                time.sleep(args.replay_speed)

    except KeyboardInterrupt:

        print("\nStopping demo...")

    print(f"{row_count} readings processed.")

    print(
        f"Final CSVs: "
        f"{RAW_CSV_PATH}, "
        f"{OUTPUT_CSV_PATH}, "
        f"{RSUL_OUTPUT_CSV_PATH}"
    )


# ============================================================================
# MAIN
# ============================================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "PulseOS live "
            "data_collection -> "
            "trend metrics -> "
            "health-intelligence -> "
            "rsul-control pipeline"
        )
    )

    parser.add_argument("--port", default=None, help="Force a specific serial port")

    parser.add_argument(
        "--load",
        choices=[
            "low",
            "normal",
            "high",
        ],
        default=None,
        help=("Send L/N/H command " "to ESP32-S3"),
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help=("Seconds between terminal " "updates. Default: 1"),
    )

    parser.add_argument(
        "--replay",
        default=None,
        help=("Replay an existing " "amr_dataset.csv"),
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help=("Run synthetic telemetry " "through the complete pipeline"),
    )

    parser.add_argument(
        "--replay-speed",
        type=float,
        default=0.3,
        help=("Seconds between replay " "rows. Default: 0.3"),
    )

    parser.add_argument(
        "--rsul-model",
        choices=[
            "linear_regression",
            "random_forest",
        ],
        default="linear_regression",
        help=("RSUL prediction model"),
    )

    parser.add_argument(
        "--rsul-threshold",
        type=float,
        default=100.0,
        help=("Latency threshold in ms " "for RSUL failure calculation"),
    )

    parser.add_argument(
        "--trend-window",
        type=int,
        default=TREND_WINDOW,
        help=("Number of recent samples " "used for trend calculations"),
    )

    args = parser.parse_args()

    # Update global trend window.
    global TREND_WINDOW

    TREND_WINDOW = max(3, args.trend_window)

    if args.replay:

        run_replay(args)

    elif args.demo:

        run_demo(args)

    else:

        run_live(args)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()
