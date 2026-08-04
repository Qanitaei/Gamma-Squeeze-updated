"""Portfolio Hedging Engine — structure recommendations + multi-objective optimize."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from gamma_squeeze.serve.schemas import HorizonForecast, PortfolioHedgeRecommendation


OptimizeObjective = Literal["Return", "Risk", "Capital Efficiency", "Margin"]

HEDGE_STRUCTURES: tuple[str, ...] = (
    "Long Calls",
    "Long Puts",
    "Covered Calls",
    "Protective Puts",
    "Debit Spreads",
    "Credit Spreads",
    "Calendar Spreads",
    "Iron Condor",
    "Collar",
    "Dynamic Delta Hedge",
    "Gamma Hedge",
    "Vega Hedge",
    "Theta Hedge",
)

OPTIMIZE_OBJECTIVES: tuple[str, ...] = (
    "Return",
    "Risk",
    "Capital Efficiency",
    "Margin",
)


@dataclass
class HedgeStructureScore:
    structure: str
    family: str
    action: str
    size_hint: str
    hedge_ratio: float
    scores: dict[str, float]  # Return / Risk / Capital Efficiency / Margin ∈ [0,1]
    composite: float
    rationale: str
    greeks_target: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioHedgePlan:
    symbol: str
    as_of: str
    recommendations: list[HedgeStructureScore]
    optimized: dict[str, Any]
    book: dict[str, float]
    data_origin: str = "alpaca"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "recommend": list(HEDGE_STRUCTURES),
            "optimize": list(OPTIMIZE_OBJECTIVES),
            "optimized": self.optimized,
            "book": self.book,
            "data_origin": self.data_origin,
            "meta": self.meta,
        }

    def as_schema_hedges(self) -> list[PortfolioHedgeRecommendation]:
        return [
            PortfolioHedgeRecommendation(
                instrument=r.structure,
                action=r.action,
                size_hint=r.size_hint,
                hedge_ratio=r.hedge_ratio,
                rationale=r.rationale,
            )
            for r in self.recommendations[:6]
        ]


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _infer_book(row: dict[str, Any], portfolio: dict[str, float] | None = None) -> dict[str, float]:
    pf = portfolio or {}
    spot = float(row.get("spot") or pf.get("spot") or 100.0)
    position = float(pf.get("position", row.get("book_position", 1.0) or 1.0))
    delta = float(pf.get("delta", row.get("dealer_delta", position) or position))
    # normalize rough book greek exposures
    return {
        "spot": spot,
        "position": position,
        "delta": float(np.tanh(delta / max(spot, 1.0))),
        "gamma": float(np.tanh(float(row.get("dealer_gamma") or 0.0) / 1e3)),
        "vega": float(np.tanh(float(row.get("dealer_vega") or 0.0) / 1e3)),
        "theta": float(np.tanh(float(row.get("dealer_theta") or 0.0) / 1e3)),
        "net_gex": float(row.get("net_gex") or 0.0),
        "iv": float(row.get("atm_iv") or 0.25),
        "stress": float(row.get("positioning_stress") or 0.0),
        "neg_gamma": float(row.get("regime_neg_gamma") or 0.0),
    }


def _forecast_signals(
    horizons: list[HorizonForecast] | None,
    row: dict[str, Any],
    forecast: dict[str, float] | None,
) -> dict[str, float]:
    if forecast:
        return {
            "probability": float(forecast.get("probability") or 0.0),
            "magnitude": float(forecast.get("magnitude") or 0.0),
            "confidence": float(forecast.get("confidence") or 0.5),
            "duration": float(forecast.get("expected_duration") or 5.0),
        }
    if horizons:
        h5 = next((h for h in horizons if h.horizon_days == 5), horizons[0])
        return {
            "probability": float(h5.squeeze_probability),
            "magnitude": float(abs(h5.expected_magnitude_pct)),
            "confidence": float(h5.confidence_score),
            "duration": float(h5.expected_duration_days),
        }
    return {
        "probability": float(row.get("positioning_stress") or 0.0) * 0.6
        + float(row.get("regime_neg_gamma") or 0.0) * 0.4,
        "magnitude": float(abs(row.get("expected_move") or 0.0)),
        "confidence": 0.45,
        "duration": 5.0,
    }


def _score_structure(
    structure: str,
    book: dict[str, float],
    sig: dict[str, float],
) -> HedgeStructureScore:
    p = sig["probability"]
    mag = sig["magnitude"] / 10.0
    conf = sig["confidence"]
    stress = book["stress"]
    neg_g = book["neg_gamma"]
    delta = book["delta"]
    iv = book["iv"]

    # Base suitability priors by structure
    priors = {
        "Long Calls": (0.75 * p + 0.25 * mag, "long_vol_up", {"delta": 0.5, "gamma": 0.4, "vega": 0.4}),
        "Long Puts": (0.55 * (1 - p) + 0.45 * stress, "long_vol_down", {"delta": -0.5, "gamma": 0.4, "vega": 0.4}),
        "Covered Calls": (0.45 * (1 - p) + 0.35 * (1 - stress) + 0.2 * iv, "income", {"delta": 0.4, "theta": 0.5}),
        "Protective Puts": (0.5 * stress + 0.3 * abs(delta) + 0.2 * p, "protect", {"delta": -0.3, "vega": 0.3}),
        "Debit Spreads": (0.65 * p + 0.2 * mag + 0.15 * conf, "defined_risk_long", {"delta": 0.35, "vega": 0.15}),
        "Credit Spreads": (0.55 * (1 - p) + 0.25 * (1 - mag) + 0.2 * (1 - stress), "defined_risk_short", {"delta": -0.2, "theta": 0.45}),
        "Calendar Spreads": (0.4 * iv + 0.35 * (1 - abs(p - 0.5)) + 0.25 * conf, "term_structure", {"vega": 0.35, "theta": 0.2}),
        "Iron Condor": (0.5 * (1 - p) + 0.3 * (1 - stress) + 0.2 * (1 - mag), "short_vol_range", {"vega": -0.45, "theta": 0.5}),
        "Collar": (0.4 * abs(delta) + 0.35 * stress + 0.25 * (1 - p), "collar", {"delta": 0.2, "vega": 0.1}),
        "Dynamic Delta Hedge": (0.55 * abs(delta) + 0.3 * stress + 0.15 * neg_g, "delta_hedge", {"delta": -delta}),
        "Gamma Hedge": (0.6 * abs(book["gamma"]) + 0.25 * neg_g + 0.15 * stress, "gamma_hedge", {"gamma": -book["gamma"]}),
        "Vega Hedge": (0.55 * abs(book["vega"]) + 0.3 * iv + 0.15 * mag, "vega_hedge", {"vega": -book["vega"]}),
        "Theta Hedge": (0.5 * abs(book["theta"]) + 0.3 * iv + 0.2 * (1 - p), "theta_hedge", {"theta": -book["theta"]}),
    }
    suitability, family, greeks = priors[structure]
    suitability = _clip01(suitability)

    # Objective scores (higher is better for all; Risk = risk-adjusted quality)
    ret = _clip01(0.55 * suitability + 0.25 * p * (1 if "long" in family or "Debit" in structure or structure == "Long Calls" else 0.4) + 0.2 * mag)
    # Risk score: higher = safer / better risk profile for the book
    if structure in ("Protective Puts", "Collar", "Dynamic Delta Hedge", "Debit Spreads"):
        risk = _clip01(0.55 + 0.35 * stress + 0.1 * conf)
    elif structure in ("Iron Condor", "Credit Spreads", "Covered Calls"):
        risk = _clip01(0.65 - 0.4 * stress - 0.25 * p)
    elif structure in ("Long Calls", "Long Puts"):
        risk = _clip01(0.35 + 0.2 * conf - 0.15 * mag)
    else:
        risk = _clip01(0.5 + 0.2 * conf - 0.15 * abs(delta))

    # Capital efficiency: spreads/condors/calendars better than naked; hedges medium
    if structure in ("Debit Spreads", "Credit Spreads", "Iron Condor", "Calendar Spreads", "Collar"):
        capital = _clip01(0.75 + 0.15 * conf)
    elif structure in ("Covered Calls", "Protective Puts"):
        capital = 0.45
    elif structure in ("Long Calls", "Long Puts"):
        capital = _clip01(0.35 + 0.2 * (1 - iv))
    else:
        capital = _clip01(0.55 + 0.2 * (1 - abs(delta)))

    # Margin: higher = lower margin burden / better margin utilization
    if structure in ("Debit Spreads", "Long Calls", "Long Puts", "Calendar Spreads"):
        margin = 0.8
    elif structure in ("Protective Puts", "Collar", "Covered Calls"):
        margin = 0.55
    elif structure in ("Credit Spreads", "Iron Condor"):
        margin = _clip01(0.4 + 0.2 * (1 - stress))
    else:
        margin = 0.6

    scores = {
        "Return": ret,
        "Risk": risk,
        "Capital Efficiency": capital,
        "Margin": margin,
    }
    # Default balanced composite
    composite = float(
        0.30 * ret + 0.30 * risk + 0.25 * capital + 0.15 * margin
    )
    size = "full" if suitability >= 0.7 else ("half" if suitability >= 0.5 else ("starter" if suitability >= 0.35 else "pass"))
    action = "buy" if structure.startswith("Long") or structure in ("Protective Puts", "Debit Spreads", "Calendar Spreads") else (
        "sell" if structure in ("Covered Calls", "Credit Spreads", "Iron Condor") else "overlay"
    )
    rationale = (
        f"{structure}: suitability={suitability:.2f}, squeeze_P={p:.2f}, "
        f"stress={stress:.2f}, δ={delta:.2f}"
    )
    return HedgeStructureScore(
        structure=structure,
        family=family,
        action=action,
        size_hint=size,
        hedge_ratio=_clip01(0.15 + 0.55 * suitability),
        scores=scores,
        composite=composite,
        rationale=rationale,
        greeks_target={k: float(v) for k, v in greeks.items()},
    )


def optimize_hedge_book(
    scored: list[HedgeStructureScore],
    *,
    objective: OptimizeObjective | str = "Risk",
    top_k: int = 4,
    max_weight: float = 0.45,
) -> dict[str, Any]:
    """Pick/weight structures to maximize a primary objective (with soft constraints)."""
    if not scored:
        return {"objective": objective, "weights": {}, "expected_scores": {}}

    obj = objective if objective in OPTIMIZE_OBJECTIVES else "Risk"
    # Soft multi-objective: primary + small blend of others
    weights_obj = {o: 0.1 for o in OPTIMIZE_OBJECTIVES}
    weights_obj[obj] = 0.7

    ranked = sorted(
        scored,
        key=lambda s: sum(weights_obj[o] * s.scores.get(o, 0.0) for o in OPTIMIZE_OBJECTIVES),
        reverse=True,
    )
    # Drop pass sizes unless nothing else
    active = [s for s in ranked if s.size_hint != "pass"] or ranked
    chosen = active[:top_k]

    raw = np.asarray([max(s.scores.get(obj, 0.0), 1e-6) * s.composite for s in chosen], dtype=float)
    # Cap single-name weight
    w = raw / raw.sum()
    w = np.minimum(w, max_weight)
    w = w / w.sum()

    alloc = {
        s.structure: {
            "weight": float(w[i]),
            "size_hint": s.size_hint,
            "action": s.action,
            "hedge_ratio": s.hedge_ratio,
            "scores": s.scores,
        }
        for i, s in enumerate(chosen)
    }
    expected = {
        o: float(sum(alloc[s.structure]["weight"] * s.scores[o] for s in chosen))
        for o in OPTIMIZE_OBJECTIVES
    }
    return {
        "objective": obj,
        "top_k": top_k,
        "weights": {k: v["weight"] for k, v in alloc.items()},
        "allocation": alloc,
        "expected_scores": expected,
        "primary_score": expected.get(obj, 0.0),
    }


def run_portfolio_hedging_engine(
    symbol: str,
    feature_row: dict[str, Any] | pd.Series,
    *,
    as_of: str | None = None,
    horizons: list[HorizonForecast] | None = None,
    forecast: dict[str, float] | None = None,
    portfolio: dict[str, float] | None = None,
    optimize_for: OptimizeObjective | str = "Risk",
    data_origin: str = "alpaca",
) -> PortfolioHedgePlan:
    row = feature_row.to_dict() if isinstance(feature_row, pd.Series) else dict(feature_row)
    as_of_s = (as_of or str(row.get("as_of") or ""))[:10]
    book = _infer_book(row, portfolio)
    sig = _forecast_signals(horizons, row, forecast)

    scored = [_score_structure(name, book, sig) for name in HEDGE_STRUCTURES]
    scored.sort(key=lambda s: s.composite, reverse=True)
    optimized = optimize_hedge_book(scored, objective=optimize_for)

    # Promote optimized structures' size hints
    opt_set = set((optimized.get("weights") or {}).keys())
    for s in scored:
        if s.structure in opt_set and s.size_hint == "pass":
            s.size_hint = "starter"

    return PortfolioHedgePlan(
        symbol=symbol.upper(),
        as_of=as_of_s,
        recommendations=scored,
        optimized=optimized,
        book=book,
        data_origin=data_origin,
        meta={
            "structures": list(HEDGE_STRUCTURES),
            "optimize_objectives": list(OPTIMIZE_OBJECTIVES),
            "signals": sig,
        },
    )


def recommend_portfolio_hedges(
    horizons: list[HorizonForecast],
    feature_row: dict[str, Any],
) -> list[PortfolioHedgeRecommendation]:
    """Backward-compatible wrapper used by SqueezeEnsemble."""
    plan = run_portfolio_hedging_engine(
        str(feature_row.get("symbol") or "UNK"),
        feature_row,
        horizons=horizons,
        optimize_for="Risk",
    )
    return plan.as_schema_hedges()


def engine_meta() -> dict[str, Any]:
    return {
        "service": "portfolio_hedging",
        "recommend": list(HEDGE_STRUCTURES),
        "optimize": list(OPTIMIZE_OBJECTIVES),
        "optimize_labels": {
            "Return": "Maximize expected payoff under squeeze path",
            "Risk": "Favor protective / defined-risk overlays",
            "Capital Efficiency": "Favor spreads / condors / calendars",
            "Margin": "Favor lower margin burden",
        },
        "score_keys": list(OPTIMIZE_OBJECTIVES),
        "data_plane": "alpaca",
    }
