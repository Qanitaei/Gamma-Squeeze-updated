"""Risk-adjusted trade recommendations from horizon forecasts."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.serve.schemas import HorizonForecast, TradeRecommendation


def recommend_trades(
    horizons: list[HorizonForecast],
    feature_row: dict[str, Any],
) -> list[TradeRecommendation]:
    """Blend model probability with positioning stress into trade structures."""
    if not horizons:
        return []

    h5 = next((h for h in horizons if h.horizon_days == 5), horizons[min(4, len(horizons) - 1)])
    h10 = next((h for h in horizons if h.horizon_days == 10), horizons[-1])
    stress = float(feature_row.get("positioning_stress") or 0.0)
    p = h5.squeeze_probability
    mag = h5.expected_magnitude_pct
    conf = h5.confidence_score

    # Simple Kelly-ish size hint from edge × confidence
    edge = max(p - 0.5, 0.0)
    kelly = edge * conf
    if kelly >= 0.15:
        size = "full"
    elif kelly >= 0.08:
        size = "half"
    elif kelly >= 0.03:
        size = "starter"
    else:
        size = "pass"

    risk = float(max(0.0, min(1.0, (1.0 - conf) * 0.6 + (1.0 - p) * 0.4)))
    out: list[TradeRecommendation] = []

    if p >= 0.55 and stress >= 0.35:
        out.append(
            TradeRecommendation(
                structure="call_debit_spread",
                direction="long_gamma_squeeze",
                size_hint=size,
                risk_score=risk,
                rationale=(
                    f"5d squeeze P={p:.2f}, expected move {mag:.1f}%, "
                    f"positioning_stress={stress:.2f}"
                ),
                horizon_days=5,
            )
        )
        out.append(
            TradeRecommendation(
                structure="long_underlying_with_put_hedge",
                direction="long",
                size_hint="half" if size == "full" else size,
                risk_score=min(1.0, risk + 0.1),
                rationale="Directional long with defined downside if dealer chase materializes",
                horizon_days=h10.horizon_days,
            )
        )
    elif p >= 0.4:
        out.append(
            TradeRecommendation(
                structure="watch_breakout_call",
                direction="conditional_long",
                size_hint="starter",
                risk_score=0.55,
                rationale=f"Elevated but not decisive squeeze odds (P={p:.2f})",
                horizon_days=5,
            )
        )
    else:
        out.append(
            TradeRecommendation(
                structure="stand_aside",
                direction="flat",
                size_hint="pass",
                risk_score=0.2,
                rationale=f"Low squeeze probability P={p:.2f}; avoid long-gamma chase",
                horizon_days=5,
            )
        )

    # Risk-adjusted short vol when negative gamma + low squeeze odds
    if float(feature_row.get("flag_fragile_gamma") or 0) >= 0.5 and p < 0.35:
        out.append(
            TradeRecommendation(
                structure="iron_condor_or_short_strangle_reduced",
                direction="short_vol",
                size_hint="starter",
                risk_score=0.7,
                rationale="Fragile gamma but low squeeze odds — mean-revert vol cautiously",
                horizon_days=5,
            )
        )
    return out
