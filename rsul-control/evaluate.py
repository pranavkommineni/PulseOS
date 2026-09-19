import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from rsul_common import (  # noqa: E402  (also lets joblib unpickle DeltaRegressor)
    DEFAULT_LAGS,
    DEFAULT_WINDOWS,
    FUTURE_COLUMNS,
    TARGETS,
    add_temporal_features,
)

DEFAULT_DATA = ROOT / "data" / "processed" / "person3_training_features.csv"

SOURCE_FOR_FUTURE = {fut: src for fut, src in TARGETS.values()}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
_frame_cache = {}


def get_frame(args, bundle):
    """Return the dataframe (with future_* columns) appropriate for this bundle."""
    if not args.raw:
        if "processed" not in _frame_cache:
            _frame_cache["processed"] = pd.read_csv(args.data)
        return _frame_cache["processed"]

    horizon = int(bundle.get("horizon", args.horizon))
    group_col = bundle.get("group_col")
    fc = bundle.get("feature_config")
    key = (horizon, group_col, json.dumps(fc, sort_keys=True, default=str))
    if key in _frame_cache:
        return _frame_cache[key]

    from train import create_targets, preprocess  # same code path as training

    df = preprocess(pd.read_csv(args.data), group_col)
    if fc:
        df, _ = add_temporal_features(
            df,
            columns=fc.get("columns"),
            lags=tuple(fc.get("lags", DEFAULT_LAGS)),
            windows=tuple(fc.get("windows", DEFAULT_WINDOWS)),
            group_col=group_col,
        )
    df = create_targets(df, horizon, group_col)
    _frame_cache[key] = df
    return df


def pick_rows(df, bundle, split):
    if split == "all":
        return df.reset_index(drop=True)

    ts = bundle.get("test_start_timestamp")
    if ts and "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        return df[parsed >= pd.Timestamp(ts)].reset_index(drop=True)

    # Older bundles: same integer arithmetic train.py used.
    n = len(df)
    start = int(
        n * (bundle.get("train_ratio", 0.70) + bundle.get("validation_ratio", 0.15))
    )
    return df.iloc[start:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def reg_scores(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def clf_scores(y_true, y_pred, proba=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }
    if proba is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
        out["pr_auc"] = float(average_precision_score(y_true, proba))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(args):
    split = args.split or ("all" if args.raw else "test")

    model_files = sorted(glob.glob(os.path.join(args.models_dir, "*.joblib")))
    if not model_files:
        print(f"No .joblib files found in {args.models_dir}")
        return

    print(
        f"Data : {args.data}  ({'raw, re-processed' if args.raw else 'processed by train.py'})"
    )
    print(f"Rows : {split.upper()} split\n")

    results = []

    for path in model_files:
        fname = os.path.basename(path)
        print(f"=== {fname} ===")
        bundle = joblib.load(path)

        if not isinstance(bundle, dict) or bundle.get("model") is None:
            print("  Not a bundle dict with a 'model' entry. Skipping.\n")
            continue

        model = bundle["model"]
        features = bundle.get("features")
        is_clf = bundle.get("task") == "classification" or "threshold_ms" in bundle

        target_col = bundle.get("target_column") or (
            "future_latency" if is_clf else None
        )
        source_col = bundle.get("source_column") or SOURCE_FOR_FUTURE.get(target_col)

        if target_col is None or source_col is None or features is None:
            print(f"  Bundle lacks target/source/features info. Keys: {list(bundle)}\n")
            continue

        df = get_frame(args, bundle)

        needed = list(features) + [target_col, source_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"  Data is missing columns this model needs: {missing}. Skipping.\n")
            continue

        rows = pick_rows(df, bundle, split)
        if rows.empty:
            print("  No rows selected for evaluation. Skipping.\n")
            continue

        X = rows[features]
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            print(
                "  Non-finite values in features. Skipping (use --raw to re-process).\n"
            )
            continue

        print(
            f"  model={bundle.get('model_name', type(model).__name__)}  "
            f"target={target_col}  horizon={bundle.get('horizon')}  "
            f"mode={bundle.get('target_mode', 'n/a')}  rows={len(rows):,}"
        )

        record = {
            "model_file": fname,
            "model_name": bundle.get("model_name", type(model).__name__),
            "target": target_col,
            "horizon": bundle.get("horizon"),
            "rows": len(rows),
            "split": split,
        }

        if not is_clf:
            y_true = rows[target_col].to_numpy(dtype=float)
            y_pred = model.predict(X)
            base = rows[source_col].to_numpy(dtype=float)
            m = reg_scores(y_true, y_pred)
            b = reg_scores(y_true, base)
            skill = 1.0 - m["rmse"] / b["rmse"]

            print(
                f"  model       R2={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}"
            )
            print(
                f"  persistence R2={b['r2']:.4f}  MAE={b['mae']:.4f}  RMSE={b['rmse']:.4f}"
            )
            print(
                f"  skill vs persistence = {skill:+.3f}  ->  "
                f"{'BEATS the baseline' if skill > 0 else 'does NOT beat the baseline'}"
            )
            record.update(
                task="regression",
                **m,
                baseline_r2=b["r2"],
                baseline_rmse=b["rmse"],
                skill_vs_persistence=skill,
                beats_baseline=skill > 0,
            )
        else:
            thr_ms = bundle.get("threshold_ms", 100.0)
            thr_p = float(bundle.get("threshold_prob", 0.5))
            y_true = (rows[target_col].to_numpy(dtype=float) >= thr_ms).astype(int)
            proba = None
            if hasattr(model, "predict_proba") and len(model.classes_) == 2:
                proba = model.predict_proba(X)[:, list(model.classes_).index(1)]
                y_pred = (proba >= thr_p).astype(int)
            else:
                y_pred = model.predict(X)

            m = clf_scores(y_true, y_pred, proba)
            persist = (rows[source_col].to_numpy(dtype=float) >= thr_ms).astype(int)
            b = clf_scores(y_true, persist)

            print(
                f"  critical if {target_col} >= {thr_ms} ms   decision threshold={thr_p:.2f}   "
                f"positives={int(y_true.sum())}/{len(y_true)} ({100 * y_true.mean():.1f}%)"
            )
            print(
                f"  model       precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
                f"F1={m['f1']:.3f}  ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}"
            )
            print(
                f"  persistence precision={b['precision']:.3f}  recall={b['recall']:.3f}  "
                f"F1={b['f1']:.3f}   (current {source_col} >= threshold)"
            )
            print(
                "  Confusion matrix [[TN FP] [FN TP]]:\n",
                confusion_matrix(y_true, y_pred, labels=[0, 1]),
            )
            record.update(
                task="classification",
                **m,
                baseline_f1=b["f1"],
                beats_baseline=m["f1"] > b["f1"],
            )

        results.append(record)
        print()

    if results:
        out_df = pd.DataFrame(results)
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_df.to_csv(args.output, index=False)
        print(f"Saved summary -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--models-dir", default=str(ROOT / "models"))
    parser.add_argument(
        "--raw",
        action="store_true",
        help="--data is a RAW file: re-run train.py's preprocessing, history "
        "features and target creation before scoring.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "all"],
        default=None,
        help="Rows to score. Default: 'test' for the processed file, 'all' for --raw.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Only used with --raw for bundles that did not save a horizon.",
    )
    parser.add_argument(
        "--output", default=str(ROOT / "results" / "saved_model_accuracy.csv")
    )
    main(parser.parse_args())
