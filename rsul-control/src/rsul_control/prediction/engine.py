"""Person 3 prediction engine.

Detailed mode uses the trained ML models. Legacy mode is a compatibility
fallback based only on the legacy contract and is explicitly labelled as such.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import joblib
import numpy as np
import pandas as pd
from functools import lru_cache

from rsul_control.adapter.person2_adapter import adapt

THRESHOLD_MS = 100.0
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
RESULTS_PATH = Path(__file__).resolve().parents[3] / "results" / "model_comparison.csv"

# Sane physical bound for degradation_rate forecasts. Linear regression can
# extrapolate without limit outside the training distribution (observed in
# production: predicted_degradation_rate of 34-59 against a training max of
# 0.697), which then poisons dominant_degradation_factor/recommendations.
# We cap at a generous multiple of the max value ever seen in training so a
# real fast-degradation event still gets through, but runaway extrapolation
# doesn't.
_DEGRADATION_TRAIN_MAX = 0.6972
DEGRADATION_RATE_CAP = _DEGRADATION_TRAIN_MAX * 5


def _best_model_per_target() -> dict[str, str]:
    """Pick the model with the highest R2 per target from the latest
    evaluation run, instead of hardcoding one model for every target.
    Falls back to linear_regression if results are unavailable.
    """
    defaults = {"latency": "linear_regression", "health": "linear_regression", "degradation": "random_forest"}
    if not RESULTS_PATH.exists():
        return defaults
    try:
        df = pd.read_csv(RESULTS_PATH)
        best = df.loc[df.groupby("target")["R2"].idxmax()]
        return {row["target"]: row["model"] for _, row in best.iterrows()}
    except Exception:
        return defaults


BEST_MODEL_FOR_TARGET = _best_model_per_target()


@lru_cache(maxsize=32)
def _load_bundle(model_name: str, target: str):
    """Load a model bundle from disk once and cache it in memory.

    Previously every single prediction re-read and re-unpickled the .joblib
    file from disk (measured: ~39ms/call, almost entirely disk I/O for 3
    files loaded on every row). Caching brings that down to a few hundred
    microseconds after the first call per model, since the process keeps the
    fitted estimator in memory instead of re-deserializing it every time.
    """
    path = MODEL_DIR / f"{model_name}_{target}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def _load_models_row(row: pd.Series, model_name: str, target: str) -> float | None:
    bundle = _load_bundle(model_name, target)
    if bundle is None:
        return None
    features = bundle["features"]
    x = row[features].to_frame().T
    value = float(bundle["model"].predict(x)[0])
    if bundle.get("predicts_delta"):
        value += float(row[bundle["current_column"]])
    return value


def _risk(prob: float) -> str:
    return "CRITICAL" if prob >= 80 else "HIGH" if prob >= 60 else "MEDIUM" if prob >= 30 else "LOW"


def _legacy_prediction(p: dict[str, Any], threshold: float) -> dict[str, Any]:
    health = float(p["shi"])
    rate = float(p["degradation_rate"])
    accel = float(p["degradation_acceleration"])

    # Legacy contract does not contain latency, so do not invent an AI latency
    # measurement. Predict time to SHI=40, the project's critical health band.
    critical_health = 40.0
    if health <= critical_health:
        rsul = 0.0
        critical_time = p["timestamp"]
    elif rate < 0:
        minutes = (health - critical_health) / abs(rate)
        rsul = minutes / 60.0
        if isinstance(p["timestamp"], (int, float)):
            critical_time = float(p["timestamp"]) + minutes
        else:
            try:
                critical_time = (pd.to_datetime(p["timestamp"]) + pd.Timedelta(minutes=minutes)).isoformat()
            except Exception:
                critical_time = None
    else:
        rsul = math.inf
        critical_time = None

    urgency = 55 * math.exp(-rsul / 4) if math.isfinite(rsul) else 0
    proximity = np.clip((100 - health) / 60, 0, 1) * 30
    accel_component = np.clip(abs(accel), 0, 5) * 3
    probability = float(np.clip(proximity + urgency + accel_component + float(p["cause_confidence"]) * 10, 0, 100))

    cause = p["root_cause"]
    recommendations = {
        "MEMORY_PRESSURE": ("Inspect memory allocation and schedule safe maintenance before critical health.", "HIGH"),
        "AI_OVERLOAD": ("Reduce AI inference load or switch to a lighter model before critical health.", "HIGH"),
        "CPU_OVERLOAD": ("Reduce competing workload and reserve CPU time for perception/control tasks.", "HIGH"),
        "DEADLINE_MISS": ("Review task priorities, scheduling intervals, and queue contention.", "CRITICAL"),
    }
    rec, priority = recommendations.get(cause, ("Continue monitoring degradation and schedule preventive maintenance.", "MEDIUM"))

    return {
        "timestamp": str(p["timestamp"]),
        "current_health": round(health, 2),
        "health_state": str(p["health_state"]),
        "current_degradation_rate": round(rate, 4),
        "predicted_health": round(max(0, health + rate * 10), 2),
        "predicted_degradation_rate": round(rate + accel * 10, 4),
        "predicted_critical_time": critical_time,
        "failure_probability": round(probability, 2),
        "risk_level": _risk(probability),
        "predicted_rsul_hours": "No predicted crossing" if not math.isfinite(rsul) else round(rsul, 3),
        "predicted_failure_time": critical_time,
        "rsul_confidence": round(float(np.clip(55 + p["cause_confidence"] * 35, 45, 90)), 1),
        "dominant_degradation_factor": cause,
        "factor_contribution": round(p["cause_confidence"] * 100, 1),
        "recommendation": rec,
        "recommendation_priority": priority,
        "compatibility_mode": "legacy",
    }


def _detailed_prediction(payload: dict[str, Any], model_name: str, threshold: float) -> dict[str, Any]:
    row = pd.Series(payload)
    numeric = [c for c in row.index if c not in {"timestamp", "health_state", "fault_type"}]
    for c in numeric:
        row[c] = pd.to_numeric(row[c], errors="coerce")
    row = row.copy()
    for c in numeric:
        if pd.isna(row[c]):
            row[c] = 0.0
    timestamp = pd.to_datetime(row["timestamp"])

    current_health = float(row["overall_health_score"])
    current_latency = float(row["latency_mean"])
    current_deg = float(row["degradation_rate"])

    # Use the best-performing model per target (per results/model_comparison.csv)
    # rather than one model_name for everything. An explicit model_name that
    # names a model with no comparison entry (or when results are missing)
    # still falls back to it directly, e.g. for manual overrides/testing.
    latency_model = BEST_MODEL_FOR_TARGET.get("latency", model_name)
    health_model = BEST_MODEL_FOR_TARGET.get("health", model_name)
    degradation_model = BEST_MODEL_FOR_TARGET.get("degradation", model_name)

    future_health_model = _load_models_row(row, health_model, "health")
    future_deg = _load_models_row(row, degradation_model, "degradation")
    future_latency = _load_models_row(row, latency_model, "latency")
    if future_deg is not None:
        future_deg = float(np.clip(future_deg, 0.0, DEGRADATION_RATE_CAP))
    horizon = 10

    # health_change_rate (dH/dt, +ve = improving) is Person 2's own live trend
    # estimate for this exact reading. Use it as a physically-grounded
    # "persistence" forecast and blend it 50/50 with the ML model's forecast.
    # This is what actually kills the "predicted_health jumps between 0 and
    # 100 for near-identical consecutive readings" bug: the ML model reacts
    # to noisy engineered features (a flag flip, a jumpy growth_rate), but
    # health_change_rate moves smoothly, so averaging the two damps model
    # noise while still tracking genuine fast degradation (which shows up in
    # health_change_rate too, so the blend isn't fighting real crashes).
    health_change_rate = float(row.get("health_change_rate", 0.0) or 0.0)
    persistence_health = float(np.clip(current_health + health_change_rate * horizon, 0, 100))
    if future_health_model is None:
        future_health = None
    else:
        future_health = float(np.clip(0.5 * future_health_model + 0.5 * persistence_health, 0, 100))

    if current_latency >= threshold:
        rsul_hours = 0.0
        critical_time = timestamp
    else:
        slope = ((future_latency - current_latency) / horizon) if future_latency is not None else float(row["latency_growth_rate"])
        if slope <= 0:
            rsul_hours = math.inf
            critical_time = None
        else:
            minutes = (threshold - current_latency) / slope
            rsul_hours = max(0.0, minutes / 60.0)
            critical_time = timestamp + pd.Timedelta(minutes=minutes)

    urgency = 0 if math.isinf(rsul_hours) else 55 * math.exp(-rsul_hours / 4)
    proximity = np.clip(current_latency / threshold, 0, 1) * 30
    trend = np.clip(max(current_deg, 0), 0, 5) * 3
    faults = min(float(row["fault_count"]), 5) * 2
    probability = float(np.clip(proximity + urgency + trend + faults, 0, 100))

    factors = {
        "AI latency": max(0.0, float(row["latency_growth_rate"])) * 1.8 + (1 if row["ai_latency_flag"] else 0),
        "CPU load": max(0.0, float(row["cpu_growth_rate"])) * 1.2 + (1 if row["cpu_overload_flag"] else 0),
        "Memory pressure": max(0.0, float(row["memory_growth_rate"])) * 1.2 + (1 if row["memory_leak_flag"] else 0),
        "Deadline degradation": max(0.0, float(row["deadline_trend"])) * 2 + (1 if row["deadline_miss_flag"] else 0),
        "Queue pressure": max(0.0, float(row["queue_growth_rate"])) * 1.2 + (1 if row["queue_overflow_flag"] else 0),
        "Stack pressure": max(0.0, float(row["stack_growth_rate"])) * 1.1 + (1 if row["stack_risk_flag"] else 0),
    }
    total = sum(factors.values()) or 1
    factor = max(factors, key=factors.get)
    contribution = round(100 * factors[factor] / total, 1)

    recs = {
        "AI latency": ("Optimize or reload the AI model; consider reducing inference frequency.", "HIGH"),
        "CPU load": ("Reduce competing workload and reserve CPU time for perception/control tasks.", "HIGH"),
        "Memory pressure": ("Inspect memory allocation and restart the affected service during a safe maintenance window.", "HIGH"),
        "Deadline degradation": ("Review FreeRTOS task priorities, queue contention, and scheduling intervals.", "CRITICAL"),
    }
    rec, priority = recs.get(factor, ("Continue monitoring runtime degradation and schedule maintenance if the predicted RSUL continues to decrease.", "MEDIUM"))

    # Confidence penalizes both a large latency jump AND a degradation forecast
    # that sits far outside the range the model was ever trained on -- the
    # latter used to slip through silently (e.g. 59.1 vs a 0.697 training max)
    # and still get reported as "45% confidence" like every other row.
    deg_extrapolation_penalty = 0.0
    if future_deg is not None and _DEGRADATION_TRAIN_MAX > 0:
        overshoot = max(0.0, future_deg - _DEGRADATION_TRAIN_MAX) / _DEGRADATION_TRAIN_MAX
        deg_extrapolation_penalty = min(30.0, overshoot * 15.0)

    # Cross-metric reconciliation: failure_probability/risk_level come from a
    # latency-driven formula, predicted_health comes from a blended ML+trend
    # forecast, and predicted_rsul_hours comes from the latency slope. These
    # used to be reported independently with no cross-check, producing rows
    # where "predicted_health: 100 (great)" and "predicted_rsul_hours: 2
    # minutes (critical)" sat side by side. When they disagree in direction,
    # trust the data less and say so via rsul_confidence rather than silently
    # reporting three contradictory numbers.
    health_delta = None if future_health is None else future_health - current_health
    disagreement_penalty = 0.0
    metrics_consistent = True
    if health_delta is not None:
        risk_says_worsening = _risk(probability) in {"HIGH", "CRITICAL"}
        risk_says_improving = _risk(probability) == "LOW"
        if risk_says_worsening and health_delta > 5:
            disagreement_penalty = 20.0
            metrics_consistent = False
        elif risk_says_improving and health_delta < -5:
            disagreement_penalty = 20.0
            metrics_consistent = False

    confidence = float(np.clip(
        92 - abs((future_latency or current_latency) - current_latency) * 2
        - (0 if current_deg > 0 else 15)
        - deg_extrapolation_penalty
        - disagreement_penalty,
        30, 95,
    ))

    return {
        "timestamp": timestamp.isoformat(),
        "current_health": round(current_health, 2),
        "health_state": str(row["health_state"]),
        "current_degradation_rate": round(current_deg, 4),
        "predicted_health": None if future_health is None else round(float(np.clip(future_health, 0, 100)), 2),
        "predicted_degradation_rate": None if future_deg is None else round(max(0.0, float(future_deg)), 4),
        "predicted_critical_time": None if critical_time is None else critical_time.isoformat(),
        "failure_probability": round(probability, 2),
        "risk_level": _risk(probability),
        "predicted_rsul_hours": "No predicted crossing" if math.isinf(rsul_hours) else round(rsul_hours, 3),
        "predicted_failure_time": None if critical_time is None else critical_time.isoformat(),
        "rsul_confidence": round(confidence, 1),
        "dominant_degradation_factor": factor,
        "factor_contribution": contribution,
        "recommendation": rec,
        "recommendation_priority": priority,
        "compatibility_mode": "detailed",
        "metrics_consistent": metrics_consistent,
    }


def predict_payload(payload: dict[str, Any], model_name="ridge", threshold=THRESHOLD_MS):
    adapted = adapt(payload)
    if adapted.mode == "legacy":
        return _legacy_prediction(adapted.payload, threshold)
    return _detailed_prediction(adapted.payload, model_name, threshold)
