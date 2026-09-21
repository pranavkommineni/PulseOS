"""
rsul_common.py
==============
Shared code for train.py and evaluate.py.

Why a separate module?
  joblib pickles a class by its import path. A class defined inside train.py and
  run with `python train.py` is stored as `__main__.DeltaRegressor`, which
  evaluate.py (or your API / ESP32 gateway) can never unpickle. Keeping it here
  fixes that, and guarantees training and evaluation build features the same way.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

# target name -> (future column created by train.py, current column it comes from)
TARGETS = {
    "latency": ("future_latency", "latency_mean"),
    "health": ("future_health", "overall_health_score"),
    "degradation_rate": ("future_degradation_rate", "degradation_rate"),
    "degradation_index": ("future_degradation_index", "degradation_index"),
}
SOURCE_COLUMNS = [src for _, src in TARGETS.values()]
FUTURE_COLUMNS = [fut for fut, _ in TARGETS.values()]

# Row counters / identifiers. A tree model happily uses these as a "clock", which
# looks great inside the training file and collapses on any new data.
ID_LIKE_COLUMNS = {
    "sample_id",
    "row_id",
    "record_id",
    "id",
    "index",
    "idx",
    "run_id",
    "device_id",
    "episode",
    "episode_id",
}

DEFAULT_LAGS = (1, 5, 10)
DEFAULT_WINDOWS = (5, 20)


# ---------------------------------------------------------------------------
# Temporal (history) features -- past data only, never the future
# ---------------------------------------------------------------------------
def add_temporal_features(
    df,
    columns=None,
    lags=DEFAULT_LAGS,
    windows=DEFAULT_WINDOWS,
    group_col=None,
):
    columns = [c for c in (columns or SOURCE_COLUMNS) if c in df.columns]
    df = df.reset_index(drop=True)

    new = {}
    for col in columns:
        s = df.groupby(group_col)[col] if group_col else df[col]
        for k in lags:
            new[f"{col}__diff{k}"] = df[col] - s.shift(k)
        for w in windows:
            mean = s.transform(lambda x, w=w: x.rolling(w, min_periods=w).mean())
            std = s.transform(lambda x, w=w: x.rolling(w, min_periods=w).std())
            new[f"{col}__rmean{w}"] = mean
            new[f"{col}__rstd{w}"] = std
            new[f"{col}__dev{w}"] = df[col] - mean

    new_df = pd.DataFrame(new, index=df.index)
    base = df.drop(columns=[c for c in new if c in df.columns])
    out = pd.concat([base, new_df], axis=1)
    out = out.dropna(subset=list(new)).reset_index(drop=True)
    return out, list(new)


# ---------------------------------------------------------------------------
# Predict the CHANGE, not the absolute value
# ---------------------------------------------------------------------------
class DeltaRegressor(RegressorMixin, BaseEstimator):
    """
    Fits  y - X[base_col]  (the change over the horizon) and returns
    X[base_col] + predicted_change.

    Why: tree models cannot predict outside the target range they saw in
    training. Degradation is a trend, so the last chronological slice is
    "further along" than anything in the training slice and trees flat-line.
    The change over 10 samples is roughly stationary, so trees handle it fine,
    and with delta = 0 the model collapses to the persistence baseline instead
    of being worse than it.
    """

    def __init__(self, estimator, base_col):
        self.estimator = estimator
        self.base_col = base_col

    def _base(self, X):
        if not hasattr(X, "columns") or self.base_col not in X.columns:
            raise ValueError(
                f"DeltaRegressor needs a DataFrame containing '{self.base_col}'."
            )
        return X[self.base_col].to_numpy(dtype=float)

    def fit(self, X, y):
        base = self._base(X)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.asarray(y, dtype=float) - base)
        return self

    def predict(self, X):
        return self._base(X) + self.estimator_.predict(X)


# ---------------------------------------------------------------------------
# Feature hygiene
# ---------------------------------------------------------------------------
def find_time_counters(df, columns):
    """Columns that only ever go up (or only ever down): row counters / clocks."""
    found = []
    for c in columns:
        v = df[c].to_numpy(dtype=float)
        if len(v) < 3:
            continue
        d = np.diff(v)
        if (d > 0).all() or (d < 0).all():
            found.append(c)
    return found
