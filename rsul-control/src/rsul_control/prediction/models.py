from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

from .feature_engineering import feature_columns


def temporal_split(df: pd.DataFrame):
    """Chronological 70/15/15 split to avoid training on future observations."""
    n = len(df)
    return df.iloc[: int(0.70 * n)], df.iloc[int(0.70 * n) : int(0.85 * n)], df.iloc[int(0.85 * n) :]


def _binary_critical_metrics(clf, Xte, yte):
    """Evaluate future criticality safely even when one class is absent in a split."""
    pred = clf.predict(Xte)
    precision, recall, f1, _ = precision_recall_fscore_support(
        yte, pred, average="binary", zero_division=0
    )

    result = {
        "Accuracy": accuracy_score(yte, pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": float("nan"),
    }

    if len(clf.classes_) == 2 and yte.nunique() == 2:
        positive_index = list(clf.classes_).index(1)
        probabilities = clf.predict_proba(Xte)[:, positive_index]
        result["ROC_AUC"] = roc_auc_score(yte, probabilities)

    return result


def train_models(df: pd.DataFrame, model_dir="models"):
    work = df.dropna().copy()
    features = feature_columns(work)
    train, val, test = temporal_split(work)

    Xtr, Xte = train[features], test[features]
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Random forests do not extrapolate well outside the training target range.
    # Predicting the future change in latency and adding it to the current latency
    # makes the RF useful for the monotonic degradation trajectory in this demo.
    y_delta_train = train["future_ai_latency_ms"] - train["ai_latency_ms"]

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=4,
            max_features=0.8,
        ),
    }

    records = []
    for name, model in models.items():
        if name == "linear_regression":
            model.fit(Xtr, train["future_ai_latency_ms"])
            pred = model.predict(Xte)
        else:
            model.fit(Xtr, y_delta_train)
            pred = test["ai_latency_ms"].to_numpy() + model.predict(Xte)

        records.append(
            {
                "model": name,
                "MAE": mean_absolute_error(test["future_ai_latency_ms"], pred),
                "RMSE": mean_squared_error(test["future_ai_latency_ms"], pred) ** 0.5,
                "R2": r2_score(test["future_ai_latency_ms"], pred),
            }
        )
        joblib.dump(
            {"model": model, "features": features, "predicts_delta": name == "random_forest"},
            Path(model_dir) / f"{name}_latency.joblib",
        )

    # For classification, predict whether the future latency will reach the
    # configurable critical threshold. This directly matches the RSUL question.
    threshold = 100.0
    ytr = (train["future_ai_latency_ms"] >= threshold).astype(int)
    yte = (test["future_ai_latency_ms"] >= threshold).astype(int)

    if ytr.nunique() >= 2:
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            max_features=0.8,
        )
        clf.fit(Xtr, ytr)
        cls_metrics = _binary_critical_metrics(clf, Xte, yte)
        joblib.dump(
            {"model": clf, "features": features, "threshold_ms": threshold},
            Path(model_dir) / "random_forest_criticality.joblib",
        )
    else:
        # The current single synthetic run reaches the critical region very late,
        # so there may be no positive training examples. Keep the pipeline honest:
        # do not fabricate classifier metrics or fit a meaningless one-class model.
        cls_metrics = {
            "Accuracy": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1": float("nan"),
            "ROC_AUC": float("nan"),
        }

    return pd.DataFrame(records), cls_metrics, test
