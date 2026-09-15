import argparse
import sys
from pathlib import Path

# Allow the script to find the src/ package automatically.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rsul_control.adapter.person2_contract import PERSON2_COLUMNS, validate_person2

MODEL_DIR = ROOT_DIR / "models"

def prepare(df, horizon):
    df = validate_person2(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in [x for x in PERSON2_COLUMNS if x not in {"timestamp", "health_state", "fault_type", "fault_severity"}]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df.copy()
    numeric_cols = [c for c in PERSON2_COLUMNS if c not in {"timestamp", "health_state", "fault_type", "fault_severity"}]
    df[numeric_cols] = df[numeric_cols].interpolate().bfill().ffill()
    df["future_latency"] = df["latency_mean"].shift(-horizon)
    df["future_health"] = df["overall_health_score"].shift(-horizon)
    df["future_degradation"] = df["degradation_rate"].shift(-horizon)
    return df.dropna(subset=["future_latency", "future_health", "future_degradation"]).reset_index(drop=True)


def split(df):
    n = len(df); a = int(.70*n); b = int(.85*n)
    return df.iloc[:a], df.iloc[b:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(ROOT_DIR / "data" / "synthetic" / "person2_dummy_training_2000.csv")
    )
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()
    raw = pd.read_csv(args.input)
    df = prepare(raw, args.horizon)
    processed_dir = ROOT_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "person3_training_features.csv", index=False)

    feature_cols = [c for c in PERSON2_COLUMNS if c not in {"timestamp", "health_state", "fault_type", "fault_severity"}]
    tr, te = split(df)
    MODEL_DIR.mkdir(exist_ok=True)
    records = []

    targets = {"latency": "future_latency", "health": "future_health", "degradation": "future_degradation"}
    for target_name, target in targets.items():
        models = {
            "linear_regression": LinearRegression(),
            # Ridge (L2-regularized linear regression) instead of plain OLS for
            # the "ridge" candidate: the 43 engineered features are heavily
            # collinear (raw signal + its trend + its growth_rate + a health
            # score derived from all of them, all fed in together), which lets
            # plain LinearRegression assign huge, unstable coefficients (we
            # measured deadline_miss_rate at +76, queue_growth_rate at -19.6).
            # A single noisy input (or a binary flag flip) then swings the
            # forecast by 50-100 points between near-identical consecutive
            # readings. Ridge shrinks correlated coefficients toward each
            # other instead of letting them fight, which stabilizes the
            # prediction without materially hurting fit quality.
            "ridge": Ridge(alpha=10.0),
            "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3, max_features=.8),
        }
        current_col = {"latency": "latency_mean", "health": "overall_health_score", "degradation": "degradation_rate"}[target_name]
        for name, model in models.items():
            if name == "random_forest":
                # RF is intentionally trained on the change over the horizon so it
                # does not fail when the test period lies outside the training range.
                y_train = tr[target] - tr[current_col]
                model.fit(tr[feature_cols], y_train)
                pred = te[current_col].to_numpy() + model.predict(te[feature_cols])
                predicts_delta = True
            else:
                model.fit(tr[feature_cols], tr[target])
                pred = model.predict(te[feature_cols])
                predicts_delta = False
            records.append({"target": target_name, "model": name, "MAE": mean_absolute_error(te[target], pred),
                            "RMSE": mean_squared_error(te[target], pred) ** .5, "R2": r2_score(te[target], pred)})
            joblib.dump({"model": model, "features": feature_cols, "target": target_name,
                         "horizon_minutes": args.horizon, "predicts_delta": predicts_delta,
                         "current_column": current_col}, MODEL_DIR / f"{name}_{target_name}.joblib")

    # Binary future criticality model. It is optional if chronological training has one class.
    ytr = (tr["future_latency"] >= 100).astype(int)
    if ytr.nunique() >= 2:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", min_samples_leaf=2)
        clf.fit(tr[feature_cols], ytr)
        joblib.dump({"model": clf, "features": feature_cols, "threshold_ms": 100.0}, MODEL_DIR / "criticality.joblib")
        yte = (te["future_latency"] >= 100).astype(int)
        pred = clf.predict(te[feature_cols])
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        metrics = {"Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred, zero_division=0),
                   "Recall": recall_score(yte, pred, zero_division=0), "F1": f1_score(yte, pred, zero_division=0), "ROC_AUC": np.nan}
        if yte.nunique() == 2:
            metrics["ROC_AUC"] = roc_auc_score(yte, clf.predict_proba(te[feature_cols])[:, 1])
    else:
        metrics = {k: np.nan for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]}

    result = pd.DataFrame(records)
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    result.to_csv(results_dir / "model_comparison.csv", index=False)
    print(result.round(3).to_string(index=False))
    print("Classification:", {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in metrics.items()})

if __name__ == "__main__":
    main()
