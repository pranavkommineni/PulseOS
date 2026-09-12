"""
Diagnostic and Explainability Engine for Person 2 Health Intelligence System.

Provides detailed mathematical and evidentiary explanations for every output metric
without contaminating or modifying the production Person 2 -> Person 3 JSON contract.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class MetricExplanation:
    """Detailed diagnostic report for a single output metric."""
    metric: str
    value: Any
    formula: str
    input_metrics_used: List[str]
    derived_features_used: List[str]
    normalized_values: Dict[str, Any]
    weights_contributions: Dict[str, Any]
    historical_window_points: int
    confidence: float
    reason: str


class DiagnosticsManager:
    """
    Constructs transparent mathematical traces and explanations for health decisions.
    """

    def __init__(self):
        self.last_explanations: Dict[str, MetricExplanation] = {}

    def record_explanation(self, explanation: MetricExplanation) -> None:
        """Store explanation for a specific metric."""
        self.last_explanations[explanation.metric] = explanation

    def get_explanation(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve diagnostic explanation for a requested metric."""
        exp = self.last_explanations.get(metric_name)
        if exp is None:
            return None
        return {
            "metric": exp.metric,
            "value": exp.value,
            "formula": exp.formula,
            "input_metrics_used": exp.input_metrics_used,
            "derived_features_used": exp.derived_features_used,
            "normalized_values": exp.normalized_values,
            "weights_contributions": exp.weights_contributions,
            "historical_window_points": exp.historical_window_points,
            "confidence": exp.confidence,
            "reason": exp.reason,
        }

    def get_all_explanations(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve all diagnostic traces from the last processed sample."""
        return {k: self.get_explanation(k) for k in self.last_explanations}
