import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ===============================================================
# PROJECT PATHS
# ===============================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA_DIR = ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

DEFAULT_INPUT = SYNTHETIC_DIR / "person2_dummy_training_2000.csv"


# ===============================================================
# IMPORTS
# ===============================================================

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from sklearn.inspection import permutation_importance

from rsul_common import (
    TARGETS,
    SOURCE_COLUMNS,
    FUTURE_COLUMNS,
    ID_LIKE_COLUMNS,
    DEFAULT_LAGS,
    DEFAULT_WINDOWS,
    DeltaRegressor,
    add_temporal_features,
    find_time_counters,
)

# ===============================================================
# CONFIGURATION
# ===============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

DEFAULT_HORIZON = 10

# Critical latency threshold.
CRITICAL_LATENCY_MS = 100.0

# Minimum rows required in each split for training to be meaningful.
MIN_SPLIT_ROWS = 5


# ===============================================================
# UTILITY
# ===============================================================


def print_header(title):
    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)


def safe_mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = np.abs(actual) > 1e-9

    if mask.sum() == 0:
        return np.nan

    return float(
        np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0
    )


def ensure_finite(df, columns, context=""):
    block = df[columns].copy()

    bad_cols = []

    for column in columns:
        values = pd.to_numeric(block[column], errors="coerce")

        if values.isna().any():
            bad_cols.append(column)
            continue

        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            bad_cols.append(column)

    if bad_cols:
        raise ValueError(
            f"Non-finite or non-numeric values found in columns "
            f"{bad_cols} "
            f"{('during ' + context) if context else ''}. "
            "Check dataset preprocessing."
        )


# ===============================================================
# LOAD DATA
# ===============================================================


def load_data(path):
    print_header("1. LOADING DATA")

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{path}\n\n" f"Expected default:\n{DEFAULT_INPUT}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset at {path} is empty (0 rows).")

    print(f"Dataset: {path}")
    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


# ===============================================================
# PREPROCESS
# ===============================================================


def preprocess(df, group_col=None):
    print_header("2. PREPROCESSING")

    df = df.copy()

    if "timestamp" not in df.columns:
        raise ValueError("Dataset does not contain 'timestamp'.")

    if group_col and group_col not in df.columns:
        raise ValueError(f"--group-col '{group_col}' is not a column in the dataset.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp"])
    sort_cols = [group_col, "timestamp"] if group_col else ["timestamp"]
    n_before = len(df)
    df = (
        df.sort_values(sort_cols, kind="stable")
        .drop_duplicates(subset=sort_cols)
        .reset_index(drop=True)
    )
    n_dup = n_before - len(df)
    print(f"Rows dropped as duplicate timestamps: {n_dup}")
    if n_before and n_dup / n_before > 0.01:
        print(
            "WARNING: more than 1% of rows were dropped as duplicate timestamps. "
            "If the file contains several devices/runs, pass --group-col <column> "
            "so they are kept and processed separately."
        )

    junk_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if junk_cols:
        print(f"Dropping junk/index columns: {junk_cols}")
        df = df.drop(columns=junk_cols)

    # -----------------------------------------------------------
    # Convert numeric columns
    # -----------------------------------------------------------

    non_numeric = {"timestamp", "condition_label", "health_state", "fault_type"}
    if group_col:
        non_numeric.add(group_col)

    for column in df.columns:
        if column not in non_numeric:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    numeric_columns = df.select_dtypes(include=np.number).columns

    if len(numeric_columns) == 0:
        raise ValueError("No numeric columns found after type coercion.")

    inf_mask = np.isinf(df[numeric_columns].to_numpy())
    n_inf = int(inf_mask.sum())
    if n_inf:
        print(f"Infinite values found and converted to NaN: {n_inf}")
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    before_missing = int(df[numeric_columns].isna().sum().sum())
    fully_missing_cols = [c for c in numeric_columns if df[c].isna().all()]
    if fully_missing_cols:
        print(
            f"WARNING: columns fully missing (no usable data): "
            f"{fully_missing_cols} - will be filled with 0.0"
        )

    df[numeric_columns] = df[numeric_columns].interpolate(
        method="linear", limit_direction="forward"
    )
    df[numeric_columns] = df[numeric_columns].bfill()
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    if int(df[numeric_columns].isna().sum().sum()):
        df[numeric_columns] = df[numeric_columns].fillna(0.0)

    after_missing = int(df[numeric_columns].isna().sum().sum())

    print(f"Missing numeric values before: {before_missing}")
    print(f"Missing numeric values after : {after_missing}")

    if after_missing:
        raise ValueError("Missing values remain after preprocessing.")

    print(f"Rows after preprocessing: {len(df)}")

    if df.empty:
        raise ValueError("No rows remain after preprocessing (timestamp parsing?).")

    return df


# ===============================================================
# CREATE FUTURE TARGETS
# ===============================================================


def create_targets(df, horizon, group_col=None):
    print_header("3. CREATING FUTURE PREDICTION TARGETS")

    df = df.copy()

    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            "Required target source columns missing:\n" + "\n".join(missing)
        )

    if horizon >= len(df):
        raise ValueError(
            f"Horizon ({horizon}) must be smaller than the number of rows "
            f"({len(df)}) after preprocessing; every future value would be "
            "undefined."
        )

    # Shift INSIDE each device/run, never across the boundary between two.
    for future_col, source_col in TARGETS.values():
        series = df.groupby(group_col)[source_col] if group_col else df[source_col]
        df[future_col] = series.shift(-horizon)

    # Remove rows at the end that do not have a future value.
    df = df.dropna(subset=FUTURE_COLUMNS).reset_index(drop=True)

    # With several groups the split must still be chronological overall.
    if group_col:
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    print(f"Prediction horizon: {horizon} samples")
    print(f"Rows after target creation: {len(df)}")

    if df.empty:
        raise ValueError(
            "No rows remain after creating future targets - reduce --horizon."
        )

    return df


# ===============================================================
# TARGET QUALITY
# ===============================================================


def validate_targets(df):
    print_header("4. TARGET QUALITY CHECK")

    valid_targets = []

    for target in FUTURE_COLUMNS:
        values = pd.to_numeric(df[target], errors="coerce")

        if values.isna().all():
            print(f"\n{target}")
            print("  INVALID TARGET: all values are NaN")
            continue

        unique_count = values.nunique()

        print(f"\n{target}")
        print(f"  Min       : {values.min():.6f}")
        print(f"  Max       : {values.max():.6f}")
        print(f"  Mean      : {values.mean():.6f}")
        print(f"  Std       : {values.std():.6f}")
        print(f"  Unique    : {unique_count}")

        if unique_count <= 1:
            print("  INVALID TARGET: constant")
        else:
            print("  VALID TARGET")
            valid_targets.append(target)

    if not valid_targets:
        raise RuntimeError("\nNo usable prediction targets exist.")

    return valid_targets


# ===============================================================
# FEATURE SELECTION
# ===============================================================


def get_features(df, extra_exclude=(), group_col=None):
    print_header("5. FEATURE SELECTION")

    forbidden_exact = {
        "timestamp",
        "health_state",
        "fault_type",
        "condition_label",
        *FUTURE_COLUMNS,
    }

    forbidden_keywords = [
        "future_",
        "target",
        "label",
        "unnamed",
    ]

    candidates = []
    id_like = []

    for column in df.columns:
        low = column.lower()

        if column in forbidden_exact or column in extra_exclude:
            continue
        if group_col and column == group_col:
            continue
        if low in ID_LIKE_COLUMNS:
            id_like.append(column)
            continue
        if any(word in low for word in forbidden_keywords):
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if df[column].nunique(dropna=True) <= 1:
            print(f"  (skipping constant feature: {column})")
            continue

        candidates.append(column)

    # A column that only ever increases is a clock in disguise
    # (uptime, cumulative counters ...). Signals we predict are exempt.
    counters = find_time_counters(
        df, [c for c in candidates if c not in SOURCE_COLUMNS]
    )
    features = [c for c in candidates if c not in counters]

    # The current value of each target is always required (delta targets,
    # persistence baseline).
    for src in SOURCE_COLUMNS:
        if src in df.columns and src not in features:
            features.append(src)

    if id_like or counters:
        print(f"Excluded as row-id / clock-like columns: {id_like + counters}")

    if not features:
        raise RuntimeError("No numeric features available.")

    print(f"Number of features: {len(features)}")
    print("\nFeatures:")
    for feature in features:
        print(f"  {feature}")

    return features


# ===============================================================
# TARGET CORRELATION CHECK
# ===============================================================


def target_correlation_check(df):
    print_header("6. TARGET CORRELATION CHECK")

    existing = [c for c in FUTURE_COLUMNS if c in df.columns]

    print(df[existing].corr().round(4).to_string())

    print("\nNote: High correlation is not automatically leakage.")
    print(
        "Leakage occurs when information unavailable at prediction"
        " time is used as an input feature."
    )


# ===============================================================
# CHRONOLOGICAL SPLIT
# ===============================================================


def chronological_split(df):
    print_header("7. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")

    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()

    for name, split in (("train", train), ("valid", valid), ("test", test)):
        if len(split) < MIN_SPLIT_ROWS:
            raise ValueError(
                f"'{name}' split has only {len(split)} rows (< {MIN_SPLIT_ROWS}). "
                "Use more data or reduce --horizon."
            )

    print(f"Total : {n}")
    print(f"Train : {len(train)}")
    print(f"Valid : {len(valid)}")
    print(f"Test  : {len(test)}")

    print("\nTime ranges:")
    print(f"TRAIN: {train['timestamp'].min()} -> {train['timestamp'].max()}")
    print(f"VALID: {valid['timestamp'].min()} -> {valid['timestamp'].max()}")
    print(f"TEST : {test['timestamp'].min()} -> {test['timestamp'].max()}")

    return train, valid, test


def range_shift_report(train, test, columns):

    print("\nRange check (test rows outside the train min/max):")
    for c in columns:
        lo, hi = train[c].min(), train[c].max()
        outside = ((test[c] < lo) | (test[c] > hi)).mean() * 100.0
        print(
            f"  {c:28s} train[{lo:10.3f}, {hi:10.3f}]  "
            f"test[{test[c].min():10.3f}, {test[c].max():10.3f}]  "
            f"outside: {outside:5.1f}%"
        )


# ===============================================================
# BUILD MODELS
# ===============================================================


def build_models():
    return {
        "linear_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=3,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def make_templates(base_models, target_mode, source_col):
    if target_mode == "delta":
        return {
            name: DeltaRegressor(model, source_col)
            for name, model in base_models.items()
        }
    return base_models


# ===============================================================
# METRICS
# ===============================================================


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    if np.unique(actual).size < 2:
        r2 = np.nan
    else:
        r2 = r2_score(actual, predicted)

    mape = safe_mape(actual, predicted)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2) if not np.isnan(r2) else np.nan,
        "MAPE_percent": float(mape) if not np.isnan(mape) else np.nan,
    }


# ===============================================================
# TRAIN AND VALIDATE
# ===============================================================


def train_and_select(
    models, train, valid, features, target_name, target_column, source_col
):
    print_header(f"8. MODEL SELECTION - {target_name}")

    X_train = train[features]
    X_valid = valid[features]

    y_train = train[target_column]
    y_valid = valid[target_column]

    ensure_finite(train, features, context=f"training data for {target_name}")
    ensure_finite(valid, features, context=f"validation data for {target_name}")

    persistence = regression_metrics(y_valid, valid[source_col])
    print(
        f"Persistence baseline (future = current) on validation: "
        f"RMSE {persistence['RMSE']:.6f}   R2 {persistence['R2']:.6f}"
    )

    records = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")

        try:
            current_model = clone(model)
            current_model.fit(X_train, y_train)
            prediction = current_model.predict(X_valid)
            metrics = regression_metrics(y_valid, prediction)
        except Exception as exc:
            print(f"  Skipped ({exc})")
            continue

        print(f"  MAE  : {metrics['MAE']:.6f}")
        print(f"  RMSE : {metrics['RMSE']:.6f}")
        print(f"  R2   : {metrics['R2']:.6f}")

        records.append(
            {"target": target_name, "model": name, "split": "validation", **metrics}
        )
        trained_models[name] = current_model

    if not trained_models:
        raise RuntimeError(f"All models failed to train for target '{target_name}'.")

    validation_results = (
        pd.DataFrame(records).sort_values("RMSE", ascending=True).reset_index(drop=True)
    )

    best_name = validation_results.iloc[0]["model"]
    best_rmse = validation_results.iloc[0]["RMSE"]

    print(f"\nSelected model: {best_name}")
    if best_rmse >= persistence["RMSE"]:
        print(
            f"WARNING: no model beats the persistence baseline on validation for "
            f"'{target_name}'. The features carry no usable signal about the "
            f"change over the horizon (or the target is noise)."
        )

    return validation_results, trained_models[best_name], best_name


# ===============================================================
# FINAL TEST
# ===============================================================


def evaluate_test(
    models,
    train,
    valid,
    test,
    features,
    target_name,
    target_column,
    selected_name,
    source_col,
):
    print_header(f"9. FINAL TEST - {target_name}")

    combined = pd.concat([train, valid], ignore_index=True)

    X_train = combined[features]
    y_train = combined[target_column]

    X_test = test[features]
    y_test = test[target_column]

    ensure_finite(combined, features, context=f"combined train+valid for {target_name}")
    ensure_finite(test, features, context=f"test data for {target_name}")

    model = clone(models[selected_name])
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    metrics = regression_metrics(y_test, prediction)

    print(f"Selected model: {selected_name}")
    print(f"MAE  : {metrics['MAE']:.6f}")
    print(f"RMSE : {metrics['RMSE']:.6f}")
    print(f"R2   : {metrics['R2']:.6f}")
    print(f"MAPE : {metrics['MAPE_percent']:.4f}%")

    baseline = test[source_col].to_numpy()
    baseline_metrics = regression_metrics(y_test, baseline)

    print("\nNaive baseline (future = current):")
    print(f"MAE  : {baseline_metrics['MAE']:.6f}")
    print(f"RMSE : {baseline_metrics['RMSE']:.6f}")
    print(f"R2   : {baseline_metrics['R2']:.6f}")

    skill = 1.0 - metrics["RMSE"] / baseline_metrics["RMSE"]
    print(f"\nSkill vs persistence (1 - RMSE/RMSE_baseline): {skill:+.3f}")

    predictions = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "target": target_name,
            "actual": y_test.to_numpy(),
            "predicted": prediction,
            "baseline_predicted": baseline,
            "error": y_test.to_numpy() - prediction,
            "absolute_error": np.abs(y_test.to_numpy() - prediction),
        }
    )

    return model, metrics, baseline_metrics, predictions, skill


# ===============================================================
# CLASSIFICATION
# ===============================================================


def _make_classifier():
    return RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _positive_proba(clf, X):
    return clf.predict_proba(X)[:, list(clf.classes_).index(1)]


def train_criticality_model(train, valid, test, features):
    print_header("10. FUTURE CRITICALITY CLASSIFICATION")

    combined = pd.concat([train, valid], ignore_index=True)

    ensure_finite(combined, features, context="criticality training data")
    ensure_finite(test, features, context="criticality test data")

    def label(frame):
        return (frame["future_latency"] >= CRITICAL_LATENCY_MS).astype(int)

    y_train, y_valid, y_test = label(train), label(valid), label(test)
    y_comb = label(combined)

    print(f"Critical threshold: {CRITICAL_LATENCY_MS} ms")
    for name, y in (("train", y_train), ("valid", y_valid), ("test", y_test)):
        print(
            f"  {name:5s}: {int(y.sum())} critical of {len(y)} ({100 * y.mean():.1f}%)"
        )

    if y_comb.nunique() < 2:
        print("\nClassification skipped: training data contains only one class.")
        return None

    prob_threshold = 0.5
    if y_train.nunique() == 2 and y_valid.nunique() == 2:
        tuner = _make_classifier().fit(train[features], y_train)
        p_valid = _positive_proba(tuner, valid[features])
        grid = np.linspace(0.05, 0.95, 91)
        f1s = [
            f1_score(y_valid, (p_valid >= t).astype(int), zero_division=0) for t in grid
        ]
        prob_threshold = float(grid[int(np.argmax(f1s))])
        print(
            f"\nDecision threshold tuned on validation: {prob_threshold:.2f} "
            f"(validation F1 {max(f1s):.4f})"
        )
    else:
        print(
            "\nThreshold tuning skipped (validation/train has a single class); using 0.5"
        )

    clf = _make_classifier().fit(combined[features], y_comb)
    probability = _positive_proba(clf, test[features])
    predicted = (probability >= prob_threshold).astype(int)

    accuracy = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted, zero_division=0)
    recall = recall_score(y_test, predicted, zero_division=0)
    f1 = f1_score(y_test, predicted, zero_division=0)

    two_classes = y_test.nunique() == 2
    roc_auc = roc_auc_score(y_test, probability) if two_classes else np.nan
    avg_prec = average_precision_score(y_test, probability) if y_test.sum() else np.nan

    persist = (test["latency_mean"] >= CRITICAL_LATENCY_MS).astype(int)
    p_precision = precision_score(y_test, persist, zero_division=0)
    p_recall = recall_score(y_test, persist, zero_division=0)
    p_f1 = f1_score(y_test, persist, zero_division=0)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"ROC-AUC  : {roc_auc:.4f}")
    if not np.isnan(avg_prec):
        print(f"PR-AUC   : {avg_prec:.4f}")
    print(
        f"\nPersistence baseline (current latency >= threshold): "
        f"precision {p_precision:.4f}  recall {p_recall:.4f}  F1 {p_f1:.4f}"
    )

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": avg_prec,
        "threshold_prob": prob_threshold,
        "baseline_Precision": p_precision,
        "baseline_Recall": p_recall,
        "baseline_F1": p_f1,
    }

    return clf, metrics, prob_threshold


# ===============================================================
# FEATURE IMPORTANCE
# ===============================================================


def calculate_importance(model, X_test, y_test, target_name):
    print_header(f"11. FEATURE IMPORTANCE - {target_name}")

    try:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="neg_root_mean_squared_error",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        importance = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )

        importance = importance.sort_values(
            "importance_mean", ascending=False
        ).reset_index(drop=True)

        print(importance.head(15).round(6).to_string(index=False))

        return importance

    except Exception as exc:
        print(f"Feature importance skipped: {exc}")
        return pd.DataFrame()


# ===============================================================
# MAIN
# ===============================================================


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe RSUL model training")

    parser.add_argument(
        "--input", default=str(DEFAULT_INPUT), help="Input Person-2 dataset"
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON, help="Future prediction horizon"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=list(TARGETS),
        default=list(TARGETS),
        help="Which targets to train (default: all).",
    )
    parser.add_argument(
        "--target-mode",
        choices=["delta", "absolute"],
        default="delta",
        help="delta = predict the change over the horizon (recommended); "
        "absolute = predict the future value directly (old behaviour).",
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Column identifying a device/run. Shifts and rolling windows are "
        "then computed inside each group.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Extra columns to drop from the feature set (ids, clocks, leaks).",
    )
    parser.add_argument(
        "--lag-cols",
        nargs="*",
        default=None,
        help="Columns to build history features for "
        "(default: latency_mean, overall_health_score, degradation_rate, degradation_index).",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Disable history features (only for A/B comparison).",
    )

    args = parser.parse_args()

    if args.horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    print_header("RSUL FINAL TRAINING PIPELINE")
    print(f"Prediction horizon: {args.horizon}")
    print(f"Input dataset: {args.input}")
    print(f"Target mode: {args.target_mode}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Load -> Preprocess -> History features -> Targets -> Features
    # -----------------------------------------------------------

    df = load_data(args.input)
    df = preprocess(df, args.group_col)

    feature_config = {
        "columns": args.lag_cols or SOURCE_COLUMNS,
        "lags": list(DEFAULT_LAGS),
        "windows": list(DEFAULT_WINDOWS),
    }

    if args.no_temporal:
        feature_config = None
        print("\nHistory features disabled (--no-temporal).")
    else:
        rows_before = len(df)
        df, temporal_cols = add_temporal_features(
            df,
            columns=feature_config["columns"],
            lags=tuple(feature_config["lags"]),
            windows=tuple(feature_config["windows"]),
            group_col=args.group_col,
        )
        print(
            f"\nHistory features added: {len(temporal_cols)} "
            f"(rows lost to warm-up: {rows_before - len(df)})"
        )

    df = create_targets(df, args.horizon, args.group_col)
    valid_targets = validate_targets(df)
    features = get_features(
        df, extra_exclude=set(args.exclude), group_col=args.group_col
    )

    target_correlation_check(df)

    processed_file = PROCESSED_DIR / "person3_training_features.csv"
    df.to_csv(processed_file, index=False)
    print(f"\nProcessed dataset saved:\n{processed_file}")

    train, valid, test = chronological_split(df)
    range_shift_report(train, test, FUTURE_COLUMNS)

    test_start = str(test["timestamp"].min())

    base_models = build_models()

    all_results = []
    all_predictions = []
    importance_files = []

    for target_name in args.targets:
        target_column, source_col = TARGETS[target_name]

        if target_column not in valid_targets:
            print(f"\nSkipping {target_name}: invalid/constant target.")
            continue

        templates = make_templates(base_models, args.target_mode, source_col)

        validation_results, _, best_model_name = train_and_select(
            models=templates,
            train=train,
            valid=valid,
            features=features,
            target_name=target_name,
            target_column=target_column,
            source_col=source_col,
        )
        all_results.append(validation_results)

        final_model, test_metrics, baseline_metrics, predictions, skill = evaluate_test(
            models=templates,
            train=train,
            valid=valid,
            test=test,
            features=features,
            target_name=target_name,
            target_column=target_column,
            selected_name=best_model_name,
            source_col=source_col,
        )

        final_record = {
            "target": target_name,
            "model": best_model_name,
            "split": "test",
            "MAE": test_metrics["MAE"],
            "RMSE": test_metrics["RMSE"],
            "R2": test_metrics["R2"],
            "MAPE_percent": test_metrics["MAPE_percent"],
            "baseline_MAE": baseline_metrics["MAE"],
            "baseline_RMSE": baseline_metrics["RMSE"],
            "baseline_R2": baseline_metrics["R2"],
            "beats_baseline": bool(test_metrics["RMSE"] < baseline_metrics["RMSE"]),
            "skill_vs_persistence": skill,
            "train_rows": len(train),
            "validation_rows": len(valid),
            "test_rows": len(test),
            "horizon": args.horizon,
            "target_mode": args.target_mode,
        }
        all_results.append(pd.DataFrame([final_record]))
        all_predictions.append(predictions)

        if args.target_mode == "delta":
            imp_model = final_model.estimator_
            imp_y = test[target_column] - test[source_col]
        else:
            imp_model = final_model
            imp_y = test[target_column]

        importance = calculate_importance(imp_model, test[features], imp_y, target_name)
        if len(importance):
            importance_file = RESULTS_DIR / f"feature_importance_{target_name}.csv"
            importance.to_csv(importance_file, index=False)
            importance_files.append(importance_file)

        model_file = MODEL_DIR / f"best_{target_name}.joblib"
        joblib.dump(
            {
                "task": "regression",
                "model": final_model,
                "features": features,
                "target": target_name,
                "target_column": target_column,
                "source_column": source_col,
                "target_mode": args.target_mode,
                "horizon": args.horizon,
                "group_col": args.group_col,
                "feature_config": feature_config,
                "test_start_timestamp": test_start,
                "train_ratio": TRAIN_RATIO,
                "validation_ratio": VALID_RATIO,
                "test_ratio": TEST_RATIO,
                "model_name": best_model_name,
            },
            model_file,
        )
        print(f"\nModel saved:\n{model_file}")

    # ===========================================================
    # SAVE MODEL COMPARISON
    # ===========================================================

    print_header("12. SAVING RESULTS")

    comparison = (
        pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    )
    comparison_file = RESULTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_file, index=False)
    print(f"Model comparison:\n{comparison_file}")

    predictions_df = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )
    predictions_file = RESULTS_DIR / "predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions:\n{predictions_file}")

    # ===========================================================
    # CRITICALITY MODEL
    # ===========================================================

    classification_result = train_criticality_model(train, valid, test, features)

    if classification_result is not None:
        classifier, classification_metrics, prob_threshold = classification_result

        classifier_file = MODEL_DIR / "future_criticality.joblib"
        joblib.dump(
            {
                "task": "classification",
                "model": classifier,
                "features": features,
                "threshold_ms": CRITICAL_LATENCY_MS,
                "threshold_prob": prob_threshold,
                "target_column": "future_latency",
                "source_column": "latency_mean",
                "horizon": args.horizon,
                "group_col": args.group_col,
                "feature_config": feature_config,
                "test_start_timestamp": test_start,
                "train_ratio": TRAIN_RATIO,
                "validation_ratio": VALID_RATIO,
                "test_ratio": TEST_RATIO,
            },
            classifier_file,
        )
        print(f"\nCriticality model saved:\n{classifier_file}")

        classification_file = RESULTS_DIR / "criticality_metrics.csv"
        pd.DataFrame([classification_metrics]).to_csv(classification_file, index=False)

    # ===========================================================
    # FINAL SUMMARY
    # ===========================================================

    print_header("FINAL RSUL TRAINING SUMMARY")

    if len(comparison):
        test_results = comparison[comparison["split"] == "test"]
        if len(test_results):
            print(
                test_results[
                    [
                        "target",
                        "model",
                        "MAE",
                        "RMSE",
                        "R2",
                        "baseline_R2",
                        "skill_vs_persistence",
                        "beats_baseline",
                    ]
                ]
                .round(5)
                .to_string(index=False)
            )

    print("\nFiles generated:")
    print(f"  {processed_file}")
    print(f"  {comparison_file}")
    print(f"  {predictions_file}")
    for file in importance_files:
        print(f"  {file}")

    print("\nTRAINING PIPELINE COMPLETE")
    print("\nIMPORTANT:")
    print(
        "R2 on absolute values is inflated for trending signals - judge a model by"
        " 'skill_vs_persistence' (must be > 0). Do not interpret a high R2 as proof"
        " of real-world ESP32-S3 performance until the model is also tested"
        " on genuinely collected ESP32-S3 telemetry."
    )


# ===============================================================
# ENTRY POINT
# ===============================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n")
        print("=" * 75)
        print("TRAINING FAILED")
        print("=" * 75)
        print(f"\nError:\n{exc}")
        print("\nCheck the dataset path, required columns, and Python dependencies.")
        raise
