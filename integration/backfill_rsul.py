"""
Backfill rsul-control predictions for an already-collected
health_intelligence_output.csv.

live_pipeline.py runs the full three-stage pipeline live/in-replay. This
script is for the case you already have a finished health_intelligence_output.csv
(e.g. from a previous run) and just want to run rsul-control's prediction
engine over it, producing rsul_control_output.csv, WITHOUT touching
amr_dataset.csv or health_intelligence_output.csv.

    health_intelligence_output.csv (47-field Person-2 contract, one row per reading)
        -> rsul_control.prediction.engine.predict_payload(...)   [rsul control]
        -> rsul_control_output.csv (RSUL / failure-probability / recommendation)

health-intelligence's `timestamp` column is elapsed seconds since the engine
started (not a calendar date), but rsul-control's prediction engine needs a
real, parseable datetime to compute a calendar predicted-critical-time. So
each row's timestamp is remapped here to a calendar time: --start-time
(default: now) plus that row's elapsed-seconds offset from the first row.
That preserves the real spacing between readings while giving rsul-control
something it can compute a future date from. The original `timestamp`
column in health_intelligence_output.csv is left untouched.

Usage:
    python integration/backfill_rsul.py
    python integration/backfill_rsul.py --input health_intelligence_output.csv --output rsul_control_output.csv
    python integration/backfill_rsul.py --model random_forest --threshold 120
    python integration/backfill_rsul.py --start-time "2026-09-15T00:00:00"
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

INTEGRATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INTEGRATION_DIR.parent
RSUL_CONTROL_SRC_DIR = PROJECT_ROOT / "rsul-control" / "src"

if str(RSUL_CONTROL_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(RSUL_CONTROL_SRC_DIR))

try:
    from rsul_control.prediction.engine import predict_payload
except ImportError as e:
    sys.exit(
        "Could not import predict_payload from rsul-control/. Run this "
        f"script from the PulseOS-main repo root, and "
        f"`pip install -r rsul-control/requirements.txt`. ({e})"
    )

RSUL_OUTPUT_KEYS = [
    "timestamp", "current_health", "health_state", "current_degradation_rate",
    "predicted_health", "predicted_degradation_rate", "predicted_critical_time",
    "failure_probability", "risk_level", "predicted_rsul_hours",
    "predicted_failure_time", "rsul_confidence", "dominant_degradation_factor",
    "factor_contribution", "recommendation", "recommendation_priority",
    "compatibility_mode",
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(PROJECT_ROOT / "health_intelligence_output.csv"))
    p.add_argument("--output", default=str(PROJECT_ROOT / "rsul_control_output.csv"))
    p.add_argument("--model", choices=["linear_regression", "random_forest"], default="linear_regression")
    p.add_argument("--threshold", type=float, default=100.0, help="Latency (ms) failure threshold")
    p.add_argument(
        "--start-time",
        default=None,
        help="Calendar timestamp (ISO format) to anchor row 0 to. Defaults to now. "
        "Subsequent rows are offset by their elapsed-seconds gap in the input file.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        sys.exit(f"{args.input} has no rows.")

    start_time = pd.to_datetime(args.start_time) if args.start_time else pd.Timestamp(datetime.now())
    first_elapsed = pd.to_numeric(df["timestamp"], errors="coerce").iloc[0]

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn model-version pickle warnings
        for _, row in df.iterrows():
            payload = row.to_dict()
            elapsed = pd.to_numeric(payload.get("timestamp"), errors="coerce")
            offset_seconds = 0.0 if pd.isna(elapsed) or pd.isna(first_elapsed) else float(elapsed - first_elapsed)
            payload["timestamp"] = (start_time + pd.Timedelta(seconds=offset_seconds)).isoformat()
            results.append(predict_payload(payload, args.model, args.threshold))

    out_df = pd.DataFrame(results, columns=RSUL_OUTPUT_KEYS)
    out_df.to_csv(args.output, index=False)
    print(f"{len(out_df)} rows: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
