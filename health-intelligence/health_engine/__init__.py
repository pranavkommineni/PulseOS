"""
Person 2: Real-time Health Intelligence System for RTOS-based Edge-AI Systems.
Processes raw runtime telemetry from Person 1 and outputs actionable health intelligence for Person 3.
"""

from .engine import HealthIntelligenceEngine
from .config import health_engineConfig
from .synthetic import SyntheticTelemetryGenerator

__all__ = [
    "HealthIntelligenceEngine",
    "health_engineConfig",
    "SyntheticTelemetryGenerator",
]

__version__ = "1.0.0"
