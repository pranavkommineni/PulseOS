"""
Live pipeline: data_collection -> JSON -> health-intelligence -> CSV.

Previously the two subsystems only talked to each other on paper:
  - data_collection/load.py read the ESP32's serial CSV lines and wrote them
    straight to amr_dataset.csv / .xlsx.
  - health-intelligence had its own engine, exercised only by
    examples/demo_stream.py against synthetic data.
Nothing actually piped live telemetry from one into the other.

This script is the real pipe between them:

    ESP32 (serial, CSV lines)
        -> parsed into a Python dict
        -> converted to a JSON object                          [data collection]
        -> saved to amr_dataset.csv (raw telemetry, as received)
        -> HealthIntelligenceEngine.process_reading(json_str)   [health intelligence]
        -> 47-key health JSON output
        -> saved to health_intelligence_output.csv (health scores/trends/faults)
        -> printed to the terminal at most once per second

JSON is what travels between data_collection and health-intelligence in
memory; two CSV files are what gets persisted to disk — one per side.

Usage:
    python integration/live_pipeline.py                  # live ESP32 over serial
    python integration/live_pipeline.py --port COM9       # force a serial port
    python integration/live_pipeline.py --fault           # also start fault injection
    python integration/live_pipeline.py --interval 2      # print every 2s instead of 1s
    python integration/live_pipeline.py --demo            # no hardware or CSV needed —
                                                            # runs the whole pipeline
                                                            # against synthetic telemetry
    python integration/live_pipeline.py --replay path/to/amr_dataset.csv
                                                            # streams an existing raw
                                                            # CSV through the same
                                                            # JSON -> engine -> CSV path

Note: the ESP32 firmware (arduino_ide.ino) only prints its CSV header line
once, right after boot. If the board was already running before this script
attached, that line is gone — this script no longer depends on catching it;
the column schema is hardcoded (HEADER_COLS below) to match the firmware
exactly, so live mode works immediately regardless of boot timing.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

INTEGRATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INTEGRATION_DIR.parent
HEALTH_INTEL_DIR = PROJECT_ROOT / "health-intelligence"

# Make `from health_engine.engine import HealthIntelligenceEngine` resolvable
# regardless of the working directory the script is run from.
if str(HEALTH_INTEL_DIR) not in sys.path:
    sys.path.insert(0, str(HEALTH_INTEL_DIR))

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

try:
    from health_engine.engine import HealthIntelligenceEngine, PRODUCTION_OUTPUT_KEYS
except ImportError as e:
    sys.exit(
        "Could not import HealthIntelligenceEngine from health-intelligence/. "
        f"Run this script from the PulseOS-main repo root. ({e})"
    )

BAUD_RATE = 115200
OUTPUT_CSV_PATH = "health_intelligence_output.csv"

# Matches the start of the header line printed by arduino_ide.ino
EXPECTED_HEADER_PREFIX = "timestamp,sample_id,uptime_ms,scenario_id"

# The firmware (Monitor_Task in arduino_ide.ino) only prints this header line
# ONCE per boot. If the ESP32 was already running before this script attached
# — or the port-open didn't trigger a hardware reset — that line is gone for
# good and will never be seen again. So the columns are hardcoded here (they
# match arduino_ide.ino's Serial.println(...) exactly) and used immediately;
# a header line on the wire is simply recognized and skipped rather than
# being the only way to learn the schema.
HEADER_COLS = [
    "timestamp", "sample_id", "uptime_ms", "scenario_id", "cpu_utilization",
    "cpu_idle", "task_cpu_utilization", "total_heap", "free_heap", "used_heap",
    "heap_utilization", "minimum_free_heap", "stack_high_water_mark",
    "stack_utilization", "task_name", "task_priority", "task_state",
    "task_execution_time", "task_period", "task_jitter", "task_execution_count",
    "deadline_misses", "context_switches", "task_switches", "scheduler_delay",
    "active_task_count", "inference_time", "min_inference_time",
    "max_inference_time", "average_inference_time", "inference_count",
    "inference_frequency", "queue_length", "queue_capacity", "queue_utilization",
    "messages_sent", "messages_received", "dropped_messages", "interrupt_count",
    "interrupt_latency", "power_consumption", "system_temperature",
    "watchdog_resets", "system_resets",
]

HEARTBEAT_SECONDS = 5.0  # print a "still waiting" message if no data arrives
RAW_CSV_PATH = "amr_dataset.csv"


# --------------------------------------------------------------------------
# Serial connection helpers (same detection logic data_collection/load.py uses)
# --------------------------------------------------------------------------

def find_esp32_port():
    """Look for a plausible ESP32 USB-serial device. Returns port name or None."""
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
    ]

    for p in ports:
        desc = f"{p.description} {p.manufacturer or ''}".upper()
        if any(k.upper() in desc for k in keywords):
            return p.device

    if len(ports) == 1:
        return ports[0].device

    return None


def try_connect(port):
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        time.sleep(2)  # allow ESP32 auto-reset + boot after opening the port
        return ser
    except Exception as e:
        print(f"Could not open {port}: {e}")
        return None


# --------------------------------------------------------------------------
# data_collection side: raw serial line -> JSON
# --------------------------------------------------------------------------

def row_to_reading(row_line, header_cols):
    """Turn one raw CSV line from the ESP32 into a reading dict."""
    values = row_line.strip().split(",")
    if len(values) != len(header_cols):
        return None
    return dict(zip(header_cols, values))


# --------------------------------------------------------------------------
# health-intelligence side: reading -> JSON -> health JSON out -> CSV rows
# --------------------------------------------------------------------------

def ensure_output_csv_header():
    if not os.path.exists(OUTPUT_CSV_PATH):
        with open(OUTPUT_CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(PRODUCTION_OUTPUT_KEYS)


def append_output_csv_row(health_json):
    with open(OUTPUT_CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([health_json.get(k, "") for k in PRODUCTION_OUTPUT_KEYS])


def ensure_raw_csv_header():
    if not os.path.exists(RAW_CSV_PATH):
        with open(RAW_CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(HEADER_COLS + ["logged_at"])


def append_raw_csv_row(reading):
    with open(RAW_CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [reading.get(k, "") for k in HEADER_COLS] + [reading.get("logged_at", "")]
        )


def process_reading(engine, reading, convert_timestamp_ms=True):
    """Convert one reading to JSON, feed it to the health engine, log both ends.

    - the raw telemetry, exactly as received, is stored in RAW_CSV_PATH
    - the health engine's output is stored in OUTPUT_CSV_PATH

    convert_timestamp_ms: the firmware's `timestamp` field is millis() since
    boot (milliseconds), but the health engine's windowing/reset logic
    expects `timestamp` in SECONDS — left unconverted, every ~1s gap between
    real readings looks like a ~1000s gap to the engine, which resets its
    history every single reading (permanent INSUFFICIENT_DATA trends). So
    real hardware readings (live serial or a replayed amr_dataset.csv) get
    converted here. Synthetic/demo data is already in seconds and should be
    passed through unconverted (convert_timestamp_ms=False).
    """
    logged = dict(reading)
    logged["logged_at"] = datetime.now().isoformat(timespec="seconds")
    ensure_raw_csv_header()
    append_raw_csv_row(logged)

    payload = dict(reading)
    if convert_timestamp_ms:
        try:
            payload["timestamp"] = float(payload["timestamp"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            pass

    health_json = engine.process_reading(json.dumps(payload))
    ensure_output_csv_header()
    append_output_csv_row(health_json)
    return health_json


def print_health_json(health_json):
    print(json.dumps(health_json))


# --------------------------------------------------------------------------
# Live (serial) mode
# --------------------------------------------------------------------------

def run_live(args):
    if serial is None:
        sys.exit("Missing dependency. Run: pip install pyserial")

    port = args.port or find_esp32_port()
    if not port:
        print("NOT LIVE: no ESP32 detected on any USB serial port.")
        print(
            "Plug in the ESP32 (flashed with arduino_ide.ino) and re-run this script, "
            "or use --replay <csv> to stream previously collected data instead."
        )
        return

    ser = try_connect(port)
    if not ser or not ser.is_open:
        print(f"NOT LIVE: found port {port} but could not open a live connection.")
        return

    print(f"LIVE: connected to ESP32 on {port} @ {BAUD_RATE} baud")
    print(
        f"data_collection -> JSON -> health-intelligence -> "
        f"{RAW_CSV_PATH} + {OUTPUT_CSV_PATH} "
        f"(terminal updates every {args.interval:g}s). Ctrl+C to stop."
    )

    if args.fault:
        ser.write(b"F\n")
        print("Sent fault-injection START command to ESP32.")

    engine = HealthIntelligenceEngine()
    row_count = 0
    malformed_count = 0
    last_print_ts = 0.0
    last_data_ts = time.monotonic()
    started_at = time.monotonic()

    try:
        while True:
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw:
                if ser.in_waiting == 0 and not ser.is_open:
                    print("NOT LIVE: serial connection lost.")
                    break
                if time.monotonic() - last_data_ts >= HEARTBEAT_SECONDS:
                    print(
                        f"... still waiting for data on {port} "
                        f"({row_count} readings so far). Is the firmware running?"
                    )
                    last_data_ts = time.monotonic()
                continue

            if raw.startswith(EXPECTED_HEADER_PREFIX):
                continue  # the header line — schema is already known, just skip it

            reading = row_to_reading(raw, HEADER_COLS)
            if reading is None:
                malformed_count += 1
                if malformed_count <= 3:
                    print("WARNING: malformed row, skipping:", raw)
                continue

            health_json = process_reading(engine, reading, convert_timestamp_ms=True)
            row_count += 1
            last_data_ts = time.monotonic()

            now = time.monotonic()
            if now - last_print_ts >= args.interval:
                print_health_json(health_json)
                last_print_ts = now

    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        if ser and ser.is_open:
            ser.close()
        if row_count == 0 and time.monotonic() - started_at < HEARTBEAT_SECONDS:
            print(
                "No readings were received. If the ESP32 was already powered on "
                "before this script started, try unplugging/replugging it (or "
                "pressing its reset button) so the firmware reboots and starts "
                "sending data after the port is open."
            )
        print(f"{row_count} readings processed. Final CSVs: {RAW_CSV_PATH}, {OUTPUT_CSV_PATH}")


# --------------------------------------------------------------------------
# Replay mode (no hardware required — streams an existing raw CSV)
# --------------------------------------------------------------------------

def run_replay(args):
    src_path = args.replay
    if not os.path.exists(src_path):
        sys.exit(
            f"Replay file not found: {src_path}\n"
            "This pipeline doesn't write amr_dataset.csv itself anymore (JSON "
            "only, in memory) — there's nothing to replay unless you already "
            "have a raw telemetry CSV from somewhere. Use --demo instead to "
            "test the full pipeline with generated synthetic data."
        )

    print(
        f"REPLAY: {src_path} -> JSON -> health-intelligence -> "
        f"{RAW_CSV_PATH} + {OUTPUT_CSV_PATH} "
        f"(terminal updates every {args.interval:g}s). Ctrl+C to stop."
    )

    engine = HealthIntelligenceEngine()
    row_count = 0
    last_print_ts = 0.0

    try:
        with open(src_path, newline="") as f:
            reader = csv.DictReader(f)
            header_cols = [c for c in reader.fieldnames if c != "logged_at"]

            for row in reader:
                reading = {k: row[k] for k in header_cols}

                health_json = process_reading(engine, reading, convert_timestamp_ms=True)
                row_count += 1

                now = time.monotonic()
                if now - last_print_ts >= args.interval:
                    print_health_json(health_json)
                    last_print_ts = now

                # pace the replay so the once-per-second terminal cadence is
                # meaningful without real hardware behind it
                if args.replay_speed > 0:
                    time.sleep(args.replay_speed)
    except KeyboardInterrupt:
        print("\nStopping replay...")

    print(f"{row_count} readings processed. Final CSVs: {RAW_CSV_PATH}, {OUTPUT_CSV_PATH}")


# --------------------------------------------------------------------------
# Demo mode (no hardware, no CSV needed — generates synthetic telemetry)
# --------------------------------------------------------------------------

def run_demo(args):
    try:
        from health_engine.synthetic import SyntheticTelemetryGenerator
    except ImportError as e:
        sys.exit(f"Could not import SyntheticTelemetryGenerator. ({e})")

    print(
        f"DEMO: synthetic telemetry -> JSON -> health-intelligence -> "
        f"{RAW_CSV_PATH} + {OUTPUT_CSV_PATH} "
        f"(terminal updates every {args.interval:g}s). Ctrl+C to stop."
    )

    engine = HealthIntelligenceEngine()
    synth = SyntheticTelemetryGenerator(seed=42)
    scenarios = [("HEALTHY", 6, 0.0), ("CPU_OVERLOAD", 8, 6.0)]
    readings = []
    for name, n, start in scenarios:
        readings.extend(synth.generate_scenario(name, num_samples=n, start_time=start))

    row_count = 0
    last_print_ts = 0.0

    try:
        for reading in readings:
            health_json = process_reading(engine, reading, convert_timestamp_ms=False)
            row_count += 1

            now = time.monotonic()
            if now - last_print_ts >= args.interval:
                print_health_json(health_json)
                last_print_ts = now

            if args.replay_speed > 0:
                time.sleep(args.replay_speed)
    except KeyboardInterrupt:
        print("\nStopping demo...")

    print(f"{row_count} readings processed. Final CSVs: {RAW_CSV_PATH}, {OUTPUT_CSV_PATH}")


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="data_collection -> JSON -> health-intelligence -> CSV pipeline for PulseOS."
    )
    parser.add_argument("--port", default=None, help="Force a specific serial port")
    parser.add_argument(
        "--fault", action="store_true", help="Send 'F' to start fault injection"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between terminal updates (default: 1.0)",
    )
    parser.add_argument(
        "--replay",
        default=None,
        help="Path to an existing amr_dataset.csv to stream through the "
        "pipeline instead of reading live serial data.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the pipeline against generated synthetic telemetry — no "
        "hardware or CSV file needed. Useful for testing the setup.",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=0.3,
        help="Seconds to sleep between replayed rows, to simulate live "
        "streaming (default: 0.3, use 0 for as-fast-as-possible)",
    )
    args = parser.parse_args()

    if args.replay:
        run_replay(args)
    elif args.demo:
        run_demo(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
