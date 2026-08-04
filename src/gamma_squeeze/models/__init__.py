"""Baseline and pluggable forecasting models."""

from gamma_squeeze.models.composite_squeeze import (
    COMPOSITE_INPUT_LABELS,
    COMPOSITE_INPUTS,
    COMPOSITE_OUTPUT_LABELS,
    COMPOSITE_OUTPUTS,
    COMPOSITE_SCORE_LABELS,
    COMPOSITE_SCORES,
    CompositeScores,
    CompositeSqueezeResult,
    run_composite_squeeze,
)

__all__ = [
    "CompositeScores",
    "CompositeSqueezeResult",
    "run_composite_squeeze",
    "COMPOSITE_INPUTS",
    "COMPOSITE_INPUT_LABELS",
    "COMPOSITE_SCORES",
    "COMPOSITE_SCORE_LABELS",
    "COMPOSITE_OUTPUTS",
    "COMPOSITE_OUTPUT_LABELS",
]
