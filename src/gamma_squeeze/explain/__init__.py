"""Model explainability — SHAP, attention, PDP, counterfactuals, decision trace."""

from gamma_squeeze.explain.attributions import explain_row
from gamma_squeeze.explain.engine import (
    EXPLAIN_COMPONENTS,
    ExplanationBundle,
    engine_meta,
    explain_prediction,
)

__all__ = [
    "EXPLAIN_COMPONENTS",
    "ExplanationBundle",
    "engine_meta",
    "explain_prediction",
    "explain_row",
]
