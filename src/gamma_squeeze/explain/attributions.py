"""Feature contribution explanations (transparent heuristic attributions)."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.serve.schemas import Explanation, HorizonForecast

# Signed weights: positive contribution increases squeeze thesis
FEATURE_WEIGHTS: dict[str, float] = {
    "positioning_stress": 0.30,
    "flag_fragile_gamma": 0.18,
    "flag_near_flip": 0.12,
    "flag_call_crowding": 0.12,
    "flag_wall_magnet": 0.08,
    "gex_per_spot": -0.10,  # more positive GEX → less squeeze fuel
    "pcr_oi": -0.08,
    "iv_minus_hv": 0.10,
    "call_bias": 0.08,
    "rvol_10d": 0.05,
}


def explain_row(
    feature_row: dict[str, Any],
    horizons: list[HorizonForecast],
    *,
    top_k: int = 6,
) -> list[Explanation]:
    scored: list[tuple[str, float, str]] = []
    for feat, weight in FEATURE_WEIGHTS.items():
        raw = feature_row.get(feat)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        # Normalize rough scales
        if feat == "gex_per_spot":
            norm = max(min(val / 1e6, 3.0), -3.0)
        elif feat in ("pcr_oi",):
            norm = val - 1.0
        elif feat == "rvol_10d":
            norm = val
        else:
            norm = val
        contrib = weight * norm
        direction = "supports_squeeze" if contrib > 0 else "against_squeeze"
        scored.append((feat, contrib, direction))

    scored.sort(key=lambda t: abs(t[1]), reverse=True)
    explanations = [
        Explanation(
            feature=feat,
            contribution=float(round(contrib, 6)),
            direction=direction,
            note=_note(feat, feature_row.get(feat)),
        )
        for feat, contrib, direction in scored[:top_k]
    ]

    if horizons:
        h5 = next((h for h in horizons if h.horizon_days == 5), horizons[0])
        explanations.append(
            Explanation(
                feature="model_horizon_5d",
                contribution=float(h5.squeeze_probability),
                direction="supports_squeeze" if h5.squeeze_probability >= 0.5 else "against_squeeze",
                note=f"Calibrated 5d squeeze probability={h5.squeeze_probability:.3f}, "
                f"confidence={h5.confidence_score:.3f}",
            )
        )
    return explanations


def _note(feat: str, value: Any) -> str:
    return f"{feat}={value}"
