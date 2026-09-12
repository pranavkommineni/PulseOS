"""Adapters for the two Person-2 -> Person-3 interfaces used by PulseOS.

Detailed mode accepts the locked 47-field Person-2 telemetry contract.
Legacy mode accepts the compact PulseOS README contract. Missing detailed
telemetry is NOT fabricated; legacy mode uses only the fields that actually
exist and returns an explicit compatibility_mode marker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .person2_contract import PERSON2_COLUMNS

LEGACY_FIELDS = {
    "timestamp", "shi", "health_state", "degradation_rate",
    "degradation_acceleration", "root_cause", "cause_confidence",
}


@dataclass
class AdaptedInput:
    mode: str
    payload: dict[str, Any]


def detect_mode(payload: Mapping[str, Any]) -> str:
    keys = set(payload.keys())
    if all(k in keys for k in PERSON2_COLUMNS):
        return "detailed"
    if keys.intersection(LEGACY_FIELDS) and {"timestamp", "shi", "health_state", "degradation_rate"}.issubset(keys):
        return "legacy"
    raise ValueError(
        "Unsupported Person-2 payload. Expected either the 47-field detailed "
        "contract or the legacy PulseOS fields: timestamp, shi, health_state, "
        "degradation_rate, degradation_acceleration, root_cause, cause_confidence."
    )


def adapt(payload: Mapping[str, Any]) -> AdaptedInput:
    mode = detect_mode(payload)
    if mode == "detailed":
        return AdaptedInput("detailed", {k: payload[k] for k in PERSON2_COLUMNS})
    return AdaptedInput("legacy", {
        "timestamp": payload["timestamp"],
        "shi": float(payload["shi"]),
        "health_state": str(payload["health_state"]),
        "degradation_rate": float(payload["degradation_rate"]),
        "degradation_acceleration": float(payload.get("degradation_acceleration", 0.0)),
        "root_cause": str(payload.get("root_cause", "UNKNOWN")),
        "cause_confidence": float(payload.get("cause_confidence", 0.0)),
    })
