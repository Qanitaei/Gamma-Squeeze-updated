"""Technical pattern recognition."""

from gamma_squeeze.patterns.recognition import (
    CANDLESTICK_PATTERNS,
    CHART_PATTERNS,
    PATTERN_BACKEND,
    PATTERN_OUTPUTS,
    detect_patterns,
)

__all__ = [
    "detect_patterns",
    "CHART_PATTERNS",
    "CANDLESTICK_PATTERNS",
    "PATTERN_OUTPUTS",
    "PATTERN_BACKEND",
]
