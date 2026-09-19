import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/processed/person3_training_features.csv"

# Put the target you want to evaluate here.
# The script will automatically check which targets exist.
POSSIBLE_TARGETS = [
    "future_degradation",
    "future_health",
    "degradation_rate",
    "degradation_index",
    "overall_health_score",
    "future_latency",
    "latency",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(DATA_FILE):
    print(f"\nERROR: Dataset not found:")
    print(DATA_FILE)
    print("\nCheck the DATA_FILE path at the top of this script.")
    exit()

df = pd.read_csv(DATA_FILE)

print("\n" + "=" * 70)
print("RSUL MODEL / DATASET EVALUATION")
print("=" * 70)

print(f"\nDataset: {DATA_FILE}")
print(f"Rows:    {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# FIND TARGETS
# ============================================================

targets = [x for x in POSSIBLE_TARGETS if x in df.columns]

if not targets:
    print("\nERROR: None of the expected target columns were found.")
    print("\nAvailable columns:")
    for c in df.columns:
        print(" ", c)
    exit()

print("\nTargets found:")
for t in targets:
    print(" ", t)


# ============================================================
# DATASET QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATASET QUALITY")
print("=" * 70)

print("\nMissing values:")

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("  No missing values.")
else:
    print(missing)


print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# TARGET STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("TARGET STATISTICS")
print("=" * 70)

for target in targets:

    values = pd.to_numeric(df[target], errors="coerce").dropna()

    print(f"\n{target}")
    print(f"  Min       : {values.min():.6f}")
    print(f"  Max       : {values.max():.6f}")
    print(f"  Mean      : {values.mean():.6f}")
    print(f"  Std       : {values.std():.6f}")
    print(f"  Unique    : {values.nunique()}")


# ============================================================
# CORRELATION / POSSIBLE SYNTHETIC LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("TARGET CORRELATION CHECK")
print("=" * 70)

numeric_df = df.select_dtypes(include=np.number)

target_corr = numeric_df[[x for x in targets if x in numeric_df.columns]].corr()

print("\n")
print(target_corr.round(4).to_string())

print("\nWARNING:")
print("Correlations extremely close to +1 or -1 may indicate")
print("that multiple target variables are generated from the")
print("same underlying synthetic progression variable.")


# ============================================================
# PROGRESSION CORRELATION CHECK
# ============================================================

df["_row_progression"] = np.arange(len(df))

print("\n" + "=" * 70)
print("PROGRESSION / TIME LEAKAGE CHECK")
print("=" * 70)

for target in targets:

    if target in numeric_df.columns:

        correlation = df["_row_progression"].corr(
            pd.to_numeric(df[target], errors="coerce")
        )

        print(f"{target:30s} correlation with row progression: " f"{correlation:.6f}")

print("\nInterpretation:")
print("Values very close to +1 or -1 mean the target strongly")
print("follows dataset order. This can make chronological")
print("prediction artificially easy.")


# ============================================================
# NAIVE BASELINE
# ============================================================

print("\n" + "=" * 70)
print("NAIVE BASELINE")
print("=" * 70)

print("""
Baseline assumption:
future value = current value

This is important because an ML model should ideally
outperform a simple baseline.
""")


# ============================================================
# RANDOM SPLIT BASELINE EVALUATION
# ============================================================

results = []

for target in targets:

    data = df.copy()

    y = pd.to_numeric(data[target], errors="coerce")

    valid = y.notna()

    data = data.loc[valid].copy()
    y = y.loc[valid]

    # Remove target and internal progression column
    X = data.drop(columns=[target, "_row_progression"], errors="ignore")

    # Keep only numeric features
    X = X.select_dtypes(include=np.number)

    # Remove any obvious future/target leakage columns
    leakage_words = ["future", "target", "label"]

    leakage_columns = [
        c for c in X.columns if any(word in c.lower() for word in leakage_words)
    ]

    if leakage_columns:

        print("\n" + "-" * 70)
        print(f"Possible leakage for target: {target}")

        for c in leakage_columns:
            print("  ", c)

        X = X.drop(columns=leakage_columns, errors="ignore")

    if X.shape[1] == 0:
        print(f"\nNo usable numeric features for {target}")
        continue

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # SIMPLE RIDGE MODEL
    # --------------------------------------------------------

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge

    model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    r2 = r2_score(y_test, prediction)

    # MAPE - only when actual != 0
    nonzero = y_test != 0

    if nonzero.sum() > 0:
        mape = (
            np.mean(np.abs((y_test[nonzero] - prediction[nonzero]) / y_test[nonzero]))
            * 100
        )
    else:
        mape = np.nan

    # --------------------------------------------------------
    # NAIVE BASELINE
    # --------------------------------------------------------

    # Try to find a reasonable current-value column
    baseline_candidates = [
        "degradation_index",
        "degradation_rate",
        "overall_health_score",
        "latency",
        "health_score",
    ]

    baseline_col = None

    for c in baseline_candidates:
        if c in data.columns and c != target:
            baseline_col = c
            break

    if baseline_col:

        baseline = pd.to_numeric(data.loc[X_test.index, baseline_col], errors="coerce")

        valid_baseline = baseline.notna()

        if valid_baseline.sum() > 0:

            baseline_actual = y_test[valid_baseline]
            baseline_prediction = baseline[valid_baseline]

            baseline_mae = mean_absolute_error(baseline_actual, baseline_prediction)

            baseline_rmse = np.sqrt(
                mean_squared_error(baseline_actual, baseline_prediction)
            )

            baseline_r2 = r2_score(baseline_actual, baseline_prediction)

        else:
            baseline_mae = np.nan
            baseline_rmse = np.nan
            baseline_r2 = np.nan

    else:

        baseline_col = "mean"

        baseline_prediction = np.full(len(y_test), y_train.mean())

        baseline_mae = mean_absolute_error(y_test, baseline_prediction)

        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_prediction))

        baseline_r2 = r2_score(y_test, baseline_prediction)

    results.append(
        {
            "target": target,
            "model": "Ridge",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE_percent": mape,
            "baseline": baseline_col,
            "baseline_MAE": baseline_mae,
            "baseline_RMSE": baseline_rmse,
            "baseline_R2": baseline_r2,
        }
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

if len(results_df) == 0:

    print("\nNo models could be evaluated.")

else:

    display_cols = ["target", "model", "MAE", "RMSE", "R2", "MAPE_percent"]

    print(results_df[display_cols].round(6).to_string(index=False))


# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

for _, row in results_df.iterrows():

    target = row["target"]
    r2 = row["R2"]
    mae = row["MAE"]
    rmse = row["RMSE"]

    print(f"\nTARGET: {target}")

    print(f"  MAE  = {mae:.6f}")
    print(f"  RMSE = {rmse:.6f}")
    print(f"  R²   = {r2:.6f}")

    if r2 >= 0.90:
        print("  R² interpretation: Very high")

    elif r2 >= 0.70:
        print("  R² interpretation: Strong")

    elif r2 >= 0.50:
        print("  R² interpretation: Moderate")

    elif r2 >= 0:
        print("  R² interpretation: Weak")

    else:
        print("  R² interpretation: Worse than mean baseline")

    if row["baseline_R2"] == row["baseline_R2"]:

        print(f"  Baseline R² = " f"{row['baseline_R2']:.6f}")

        if r2 > row["baseline_R2"]:
            print("  ✓ Model beats baseline")

        else:
            print("  ⚠ Model does NOT beat baseline")


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

output_file = "results/fresh_model_evaluation.csv"

results_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"\nResults saved to:")
print(output_file)

print("\nThis evaluation is independent of your existing")
print("model_comparison.csv.")
