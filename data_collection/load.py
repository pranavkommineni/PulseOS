import argparse
import csv
import os
import sys
import time
from datetime import datetime

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("Missing dependency. Run: pip install pyserial pandas openpyxl")

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing dependency. Run: pip install pyserial pandas openpyxl")

BAUD_RATE = 115200
CSV_PATH = "amr_dataset.csv"
XLSX_PATH = "amr_dataset.xlsx"
XLSX_REWRITE_EVERY_N_ROWS = 15

# Matches the start of the header line printed by esp32_amr_monitor.ino
EXPECTED_HEADER_PREFIX = "timestamp,sample_id,uptime_ms,scenario_id"


def find_esp32_port():
    """Look for a plausible ESP32 USB-serial device. Returns port name or None."""
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


def ensure_csv_header(header_line):
    file_exists = os.path.exists(CSV_PATH)
    if not file_exists:
        with open(CSV_PATH, "w", newline="") as f:
            f.write(header_line.strip() + ",logged_at\n")


def append_row(row_line, header_cols):
    logged_at = datetime.now().isoformat(timespec="seconds")

    values = row_line.strip().split(",")

    if len(values) != len(header_cols):
        print("WARNING: Invalid CSV row, skipping:", row_line)
        return

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(values + [logged_at])


def rewrite_xlsx():
    if not os.path.exists(CSV_PATH):
        print("(warning) no CSV data collected yet.")
        return

    try:
        df = pd.read_csv(CSV_PATH)
        df.to_excel(XLSX_PATH, index=False, sheet_name="amr_metrics")
    except Exception as e:
        print(f"(warning) could not refresh xlsx: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None, help="Force a specific serial port")
    parser.add_argument(
        "--load",
        choices=["low", "normal", "high"],
        default=None,
        help="Send 'L'/'N'/'H' to set the ESP32's load level on startup",
    )
    parser.add_argument(
        "--retry-seconds",
        type=int,
        default=5,
        help="How often to retry detecting the ESP32 if not found",
    )
    args = parser.parse_args()

    port = args.port or find_esp32_port()

    if not port:
        print("NOT LIVE: no ESP32 detected on any USB serial port.")
        print(
            "Plug in the ESP32 (flashed with esp32_amr_monitor.ino) and re-run this script."
        )
        return

    ser = try_connect(port)
    if not ser or not ser.is_open:
        print(f"NOT LIVE: found port {port} but could not open a live connection.")
        return

    print(f"LIVE: connected to ESP32 on {port} @ {BAUD_RATE} baud")
    print(
        f"Logging to {CSV_PATH} (and refreshing {XLSX_PATH} every {XLSX_REWRITE_EVERY_N_ROWS} rows). "
        f"3 rows are written per monitoring cycle (one per task). Ctrl+C to stop."
    )

    if args.load:
        cmd = {"low": b"L\n", "normal": b"N\n", "high": b"H\n"}[args.load]
        ser.write(cmd)
        print(f"Sent load_level={args.load.upper()} command to ESP32.")

    header_cols = None
    row_count = 0

    try:
        while True:
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw:
                if ser.in_waiting == 0 and not ser.is_open:
                    print("NOT LIVE: serial connection lost.")
                    break
                continue

            if raw.startswith(EXPECTED_HEADER_PREFIX):
                header_cols = raw.split(",")
                ensure_csv_header(raw)
                continue

            if header_cols is None:
                # Skip any boot noise printed before the CSV header
                continue

            append_row(raw, header_cols)
            row_count += 1
            print(f"[row {row_count}] {raw}")

            if row_count % XLSX_REWRITE_EVERY_N_ROWS == 0:
                rewrite_xlsx()

    except KeyboardInterrupt:
        print("\nStopping logger...")
    finally:
        rewrite_xlsx()
        if ser and ser.is_open:
            ser.close()
        print(f"Final dataset saved: {CSV_PATH} and {XLSX_PATH}")


if __name__ == "__main__":
    main()
