import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rsul_control.prediction.engine import predict_payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default=str(ROOT_DIR / "data" / "synthetic" / "person2_dummy_training_2000.csv")
    )
    p.add_argument("--model", choices=["linear_regression", "random_forest"], default="linear_regression")
    p.add_argument("--threshold", type=float, default=100.0)
    p.add_argument("--row", type=int, default=-1)
    a = p.parse_args()
    df = pd.read_csv(a.input)
    payload = df.iloc[a.row].to_dict()
    print(json.dumps(predict_payload(payload, a.model, a.threshold), indent=2))

if __name__ == "__main__":
    main()
