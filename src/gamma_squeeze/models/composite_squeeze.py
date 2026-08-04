"""Composite Gamma Squeeze Engine — blends HMM, XGB, TFT, dealer, patterns, macro, sentiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gamma_squeeze.config import HORIZONS, alpaca_configured
from gamma_squeeze.dealer.hedge_demand import simulate_dealer_hedging
from gamma_squeeze.deep.temporal_model import TFTForecastModel, load_tft_model
from gamma_squeeze.features.macro_features import compute_macro_features
from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv
from gamma_squeeze.models.ensemble import SqueezeEnsemble, blend_confidence
from gamma_squeeze.models.xgboost_direction import load_direction_model
from gamma_squeeze.patterns.recognition import detect_patterns
from gamma_squeeze.regime.hmm_regime import fit_hmm_regimes, infer_regime, load_hmm_model
from gamma_squeeze.serve.schemas import SCHEMA_VERSION


SQUEEZE_REGIMES = {
    "Gamma Squeeze",
    "Short Squeeze",
    "Negative Gamma",
    "Gamma Expansion",
    "Panic",
    "High Volatility",
}

BREAKOUT_PATTERNS = {
    "range_breakout",
    "Bull Flag",
    "Ascending Triangle",
    "Cup Handle",
    "Double Bottom",
    "Inverse Head Shoulders",
    "Pennant",
    "Channel",
}

COMPOSITE_INPUTS = (
    "hmm",
    "xgboost",
    "tft",
    "dealer",
    "patterns",
    "macro",
    "sentiment",
)

COMPOSITE_INPUT_LABELS = {
    "hmm": "Hidden Markov",
    "xgboost": "XGBoost",
    "tft": "Temporal Transformer",
    "dealer": "Dealer Simulation",
    "patterns": "Pattern Recognition",
    "macro": "Macro",
    "sentiment": "Sentiment",
}

COMPOSITE_SCORES = (
    "gamma_squeeze_score",
    "gamma_expansion_score",
    "dealer_flow_score",
    "momentum_score",
    "liquidity_score",
    "breakout_score",
)

COMPOSITE_SCORE_LABELS = {
    "gamma_squeeze_score": "Gamma Squeeze Score",
    "gamma_expansion_score": "Gamma Expansion Score",
    "dealer_flow_score": "Dealer Flow Score",
    "momentum_score": "Momentum Score",
    "liquidity_score": "Liquidity Score",
    "breakout_score": "Breakout Score",
}

COMPOSITE_OUTPUTS = (
    "probability",
    "magnitude",
    "expected_duration",
    "expected_start",
    "confidence",
    "risk_rating",
)

COMPOSITE_OUTPUT_LABELS = {
    "probability": "Probability",
    "magnitude": "Magnitude",
    "expected_duration": "Expected Duration",
    "expected_start": "Expected Start",
    "confidence": "Confidence",
    "risk_rating": "Risk Rating",
}


@dataclass
class CompositeScores:
    gamma_squeeze_score: float = 0.0
    gamma_expansion_score: float = 0.0
    dealer_flow_score: float = 0.0
    momentum_score: float = 0.0
    liquidity_score: float = 0.0
    breakout_score: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class CompositeSqueezeResult:
    symbol: str
    as_of: str
    scores: CompositeScores = field(default_factory=CompositeScores)
    probability: float = 0.0
    magnitude: float = 0.0  # pct
    expected_duration: float = 0.0  # days
    expected_start: float = 0.0  # days ahead (0 = underway / imminent)
    confidence: float = 0.0
    risk_rating: str = "Low"
    inputs: dict[str, Any] = field(default_factory=dict)
    forecast: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of,
            "scores": self.scores.to_dict(),
            "score_labels": dict(COMPOSITE_SCORE_LABELS),
            "probability": self.probability,
            "magnitude": self.magnitude,
            "expected_duration": self.expected_duration,
            "expected_start": self.expected_start,
            "confidence": self.confidence,
            "risk_rating": self.risk_rating,
            "inputs": self.inputs,
            "input_labels": dict(COMPOSITE_INPUT_LABELS),
            "outputs": list(COMPOSITE_OUTPUTS),
            "output_labels": dict(COMPOSITE_OUTPUT_LABELS),
            "final_output": {
                "probability": self.probability,
                "magnitude": self.magnitude,
                "expected_duration": self.expected_duration,
                "expected_start": self.expected_start,
                "confidence": self.confidence,
                "risk_rating": self.risk_rating,
            },
            "forecast": self.forecast,
            "meta": self.meta,
        }


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return default if default is not None else {"error": str(exc)}


def _sentiment_score(macro: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Blend consumer sentiment + positioning/PCR into [0,1] risk-on sentiment."""
    umc = macro.get("consumer_sentiment")
    # UMCSENT typically ~50–100; map to 0–1 around 50–100
    cons = None
    if umc is not None:
        cons = _clip01((float(umc) - 40.0) / 60.0)
    pcr = float(row.get("pcr_oi") or row.get("put_call_ratio") or 1.0)
    # high PCR → fear → lower sentiment
    pcr_sent = _clip01(1.2 - 0.5 * pcr)
    stress = float(row.get("positioning_stress") or 0.0)
    # for squeeze, stressed sentiment can fuel upside — keep raw market sentiment separate
    market = _clip01(0.6 * (cons if cons is not None else 0.5) + 0.4 * pcr_sent)
    squeeze_fuel = _clip01(0.55 * stress + 0.45 * (1.0 - market))
    return {
        "market_sentiment": market,
        "squeeze_sentiment": squeeze_fuel,
        "consumer_sentiment_raw": float(umc) if umc is not None else None,
        "pcr": pcr,
    }


def _collect_inputs(
    symbol: str,
    features: pd.DataFrame,
    matrix: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Gather HMM / XGB / TFT / dealer / patterns / macro / sentiment blocks."""
    row = features.sort_values("as_of").iloc[-1].to_dict()
    as_of = str(row.get("as_of"))[:10]
    inputs: dict[str, Any] = {"as_of": as_of}
    errors: dict[str, str] = {}

    # HMM
    def _hmm():
        model = load_hmm_model() or fit_hmm_regimes(features)
        return infer_regime(features, model)

    hmm = _safe(_hmm, {})
    if isinstance(hmm, dict) and hmm.get("error"):
        errors["hmm"] = str(hmm["error"])
    inputs["hmm"] = {
        "current_state": (hmm or {}).get("current_state") or (hmm or {}).get("regime"),
        "confidence": (hmm or {}).get("confidence"),
        "next_state_probability": (hmm or {}).get("next_state_probability"),
    }

    # XGBoost direction
    def _xgb():
        bundle = load_direction_model()
        if bundle is None:
            return {"direction": "flat", "probability": {}}
        return bundle.predict_row(row)

    xgb = _safe(_xgb, {})
    inputs["xgboost"] = {
        "direction": (xgb or {}).get("direction") or (xgb or {}).get("classification"),
        "probability": (xgb or {}).get("probability") or (xgb or {}).get("proba") or {},
        "horizons": (xgb or {}).get("horizons"),
        "regression_5d": ((xgb or {}).get("horizons") or {}).get("5d", {}).get("regression")
        if isinstance((xgb or {}).get("horizons"), dict)
        else (xgb or {}).get("regression"),
    }

    # TFT
    def _tft():
        model = load_tft_model() or TFTForecastModel(lookback=min(32, max(12, len(features) // 3)))
        if not getattr(model, "models", None):
            model.feature_columns = [
                c for c in features.columns if c not in ("symbol", "as_of")
            ]
            model.quantile_mode = "median_spread"
            model.fit(features, pd.DataFrame())
        return model.forecast(features)

    tft = _safe(_tft, {})
    h5_tft = ((tft or {}).get("horizons") or {}).get("5d") or {}
    # Support nested targets block from Phase-7 TFT payload
    fp5 = h5_tft.get("future_price") or (h5_tft.get("targets") or {}).get("future_price")
    gex5 = h5_tft.get("net_gex") or (h5_tft.get("targets") or {}).get("net_gex")
    inputs["tft"] = {
        "version": (tft or {}).get("version"),
        "future_price_5d": fp5,
        "net_gex_5d": gex5,
        "attention_top": ((tft or {}).get("attention_weights") or {}).get("variables", [])[:5],
    }

    # Dealer simulation
    def _dealer():
        if not matrix:
            return {"error": "no_matrix"}
        return simulate_dealer_hedging(matrix, as_of=as_of, n_points=15).to_dict()

    dealer = _safe(_dealer, {})
    inputs["dealer"] = {
        "delta_hedging": (dealer or {}).get("delta_hedging"),
        "dealer_exhaustion": (dealer or {}).get("dealer_exhaustion"),
        "dealer_liquidity": (dealer or {}).get("dealer_liquidity"),
        "gamma_ramp": (dealer or {}).get("gamma_ramp"),
        "expected_hedging_volume": (dealer or {}).get("expected_hedging_volume"),
        "position_flip": (dealer or {}).get("dealer_position_flip"),
        "share_purchases": (dealer or {}).get("dealer_share_purchases"),
        "share_sales": (dealer or {}).get("dealer_share_sales"),
    }

    # Patterns
    def _patterns():
        ohlcv = fetch_daily_ohlcv(symbol)
        return detect_patterns(ohlcv, lookback=80, min_confidence=0.35)

    patterns = _safe(_patterns, {"patterns": []})
    top_patterns = (patterns or {}).get("patterns") or []
    inputs["patterns"] = {
        "n_patterns": len(top_patterns),
        "top": [
            {
                "pattern": p.get("pattern"),
                "confidence": p.get("confidence"),
                "probability": p.get("probability"),
            }
            for p in top_patterns[:5]
        ],
        "compression": ((patterns or {}).get("summary") or {}).get("compression"),
    }

    # Macro
    macro_feats = _safe(lambda: compute_macro_features(as_of=as_of, include_external_vol=False).flat_features(), {})
    inputs["macro"] = {
        k: (macro_feats or {}).get(k)
        for k in (
            "fed_funds",
            "cpi",
            "yield_spread",
            "vix",
            "credit_spread",
            "high_yield_spread",
            "economic_surprise_index",
            "consumer_sentiment",
            "dollar_index",
        )
    }

    # Sentiment
    inputs["sentiment"] = _sentiment_score(macro_feats or {}, row)
    inputs["feature_row"] = {
        "spot": row.get("spot"),
        "net_gex": row.get("net_gex"),
        "positioning_stress": row.get("positioning_stress"),
        "regime_neg_gamma": row.get("regime_neg_gamma"),
        "rvol_10d": row.get("rvol_10d"),
        "ret_5d": row.get("ret_5d"),
    }
    return inputs, errors


def _compute_scores(inputs: dict[str, Any]) -> CompositeScores:
    row = inputs.get("feature_row") or {}
    hmm = inputs.get("hmm") or {}
    xgb = inputs.get("xgboost") or {}
    dealer = inputs.get("dealer") or {}
    patterns = inputs.get("patterns") or {}
    macro = inputs.get("macro") or {}
    sentiment = inputs.get("sentiment") or {}

    stress = float(row.get("positioning_stress") or 0.0)
    neg_g = float(row.get("regime_neg_gamma") or 0.0)
    net_gex = float(row.get("net_gex") or 0.0)
    # normalize gex magnitude softly
    gex_neg = _clip01(-np.tanh(net_gex / 1e6))

    regime = str(hmm.get("current_state") or "")
    hmm_boost = 0.85 if regime in SQUEEZE_REGIMES else (0.55 if "Gamma" in regime else 0.35)
    hmm_conf = float(hmm.get("confidence") or 0.5)

    up_p = float((xgb.get("probability") or {}).get("up") or 0.33)
    ret5 = float(row.get("ret_5d") or 0.0)
    momentum = _clip01(0.55 * up_p + 0.45 * (0.5 + np.tanh(ret5 * 20) / 2))

    exhaustion = float(dealer.get("dealer_exhaustion") or 0.0)
    liq = float(dealer.get("dealer_liquidity") or row.get("dealer_liquidity_score") or 0.5)
    purchases = float(dealer.get("share_purchases") or 0.0)
    sales = float(dealer.get("share_sales") or 0.0)
    flow_tot = purchases + sales
    flow_score = _clip01(0.4 * exhaustion + 0.3 * min(flow_tot / 1e6, 1.0) + 0.3 * (1.0 - abs(purchases - sales) / (flow_tot + 1e-9)))

    ramp = abs(float(dealer.get("gamma_ramp") or 0.0))
    expansion = _clip01(0.5 * min(ramp / 1e5, 1.0) + 0.3 * gex_neg + 0.2 * float(bool(patterns.get("compression"))))

    breakout = 0.0
    for p in patterns.get("top") or []:
        if p.get("pattern") in BREAKOUT_PATTERNS:
            breakout = max(breakout, float(p.get("confidence") or 0))
    if patterns.get("compression"):
        breakout = max(breakout, 0.45)

    vix = macro.get("vix")
    vix_term = _clip01(((float(vix) if vix is not None else 18.0) - 12.0) / 25.0)

    squeeze = _clip01(
        0.22 * stress
        + 0.18 * neg_g
        + 0.15 * gex_neg
        + 0.15 * (hmm_boost * hmm_conf)
        + 0.12 * float(sentiment.get("squeeze_sentiment") or 0)
        + 0.10 * exhaustion
        + 0.08 * vix_term
    )

    return CompositeScores(
        gamma_squeeze_score=squeeze,
        gamma_expansion_score=expansion,
        dealer_flow_score=flow_score,
        momentum_score=momentum,
        liquidity_score=_clip01(liq),
        breakout_score=_clip01(breakout),
    )


def _risk_rating(probability: float, exhaustion: float, magnitude: float, confidence: float) -> str:
    score = 0.35 * probability + 0.25 * exhaustion + 0.25 * min(abs(magnitude) / 10.0, 1.0) + 0.15 * confidence
    if score >= 0.75:
        return "Extreme"
    if score >= 0.55:
        return "High"
    if score >= 0.35:
        return "Medium"
    return "Low"


def run_composite_squeeze(
    symbol: str,
    features: pd.DataFrame,
    matrix: dict[str, Any] | None = None,
    *,
    persist_ensemble: bool = False,
) -> CompositeSqueezeResult:
    """Build composite scores + final squeeze prognosis."""
    del persist_ensemble  # reserved
    sym = symbol.upper()
    if features.empty:
        return CompositeSqueezeResult(symbol=sym, as_of="", meta={"error": "no_features"})

    row = features.sort_values("as_of").iloc[-1]
    as_of = str(row["as_of"])[:10]
    inputs, errors = _collect_inputs(sym, features, matrix)
    scores = _compute_scores(inputs)

    # Baseline ensemble (trained models if present)
    ensemble = SqueezeEnsemble.load_latest()
    forecast = ensemble.predict(row, symbol=sym, as_of=as_of, matrix=matrix)
    fc = forecast.to_dict()
    # Prefer 5d horizon as primary
    h5 = next((h for h in fc.get("horizons") or [] if h.get("horizon_days") == 5), None)
    if h5 is None and fc.get("horizons"):
        h5 = fc["horizons"][min(4, len(fc["horizons"]) - 1)]
    base_p = float((h5 or {}).get("squeeze_probability") or 0.0)
    base_mag = float((h5 or {}).get("expected_magnitude_pct") or 0.0)
    base_dur = float((h5 or {}).get("expected_duration_days") or 5.0)
    base_conf = float((h5 or {}).get("confidence_score") or 0.0)

    # Composite probability
    s = scores
    regime = str((inputs.get("hmm") or {}).get("current_state") or "")
    regime_boost = 0.12 if regime in SQUEEZE_REGIMES else 0.0
    probability = _clip01(
        0.28 * base_p
        + 0.22 * s.gamma_squeeze_score
        + 0.14 * s.gamma_expansion_score
        + 0.12 * s.dealer_flow_score
        + 0.10 * s.momentum_score
        + 0.08 * s.breakout_score
        + 0.06 * (1.0 - abs(s.liquidity_score - 0.55))  # mid-high liquidity helps
        + regime_boost
    )

    # Magnitude: blend model with stress / expansion
    magnitude = float(
        max(
            0.0,
            0.55 * abs(base_mag)
            + 0.25 * (10.0 * s.gamma_squeeze_score)
            + 0.20 * (8.0 * s.gamma_expansion_score),
        )
    )

    # Duration
    expected_duration = float(
        max(1.0, 0.6 * base_dur + 0.4 * (2.0 + 6.0 * s.gamma_squeeze_score))
    )

    # Expected start: 0 if already in squeeze-like regime / breakout; else delay
    if regime in SQUEEZE_REGIMES or s.breakout_score >= 0.65 or s.gamma_squeeze_score >= 0.7:
        expected_start = 0.0
    else:
        expected_start = float(max(0.0, 3.0 * (1.0 - s.gamma_squeeze_score) + 1.0 * (1.0 - s.momentum_score)))

    hmm_block = inputs.get("hmm") or {}
    xgb_block = inputs.get("xgboost") or {}
    regime_state = str(hmm_block.get("current_state") or "")
    hmm_conf = float(hmm_block.get("confidence") or 0.5)
    squeeze_friendly = 1.0 if regime_state in SQUEEZE_REGIMES else (
        0.65 if "Gamma" in regime_state else 0.35
    )
    regime_agree = _clip01(squeeze_friendly * (0.45 + 0.55 * hmm_conf))
    up_p = float((xgb_block.get("probability") or {}).get("up") or 0.33)
    direction_label = str(xgb_block.get("direction") or "flat")
    if direction_label == "up":
        dir_agree = up_p
    elif direction_label == "down":
        dir_agree = _clip01(1.0 - up_p)
    else:
        dir_agree = _clip01(0.45 + 0.2 * up_p)

    # Prefer ensemble blend (HMM + XGB agreement) then fold in composite pattern support
    ensemble_conf = blend_confidence(
        probability,
        row.to_dict(),
        regime={
            "state": regime_state,
            "confidence": hmm_conf,
            "agreement": regime_agree,
            "source": "hmm" if regime_state else "fallback",
        },
        direction={
            "direction": direction_label,
            "up_probability": up_p,
            "agreement": dir_agree,
            "source": "xgb" if xgb_block.get("probability") else "fallback",
        },
        horizon=int((h5 or {}).get("horizon_days") or 5),
    )
    confidence = _clip01(
        0.70 * ensemble_conf
        + 0.15 * base_conf
        + 0.15 * min(1.0, (inputs.get("patterns") or {}).get("n_patterns", 0) / 5.0)
    )

    # Propagate blended confidence onto primary forecast horizons for schema consumers
    for hz in fc.get("horizons") or []:
        if isinstance(hz, dict):
            hz_p = float(hz.get("squeeze_probability") or 0.0)
            hz["confidence_score"] = blend_confidence(
                hz_p,
                row.to_dict(),
                regime={
                    "state": regime_state,
                    "confidence": hmm_conf,
                    "agreement": regime_agree,
                    "source": "hmm" if regime_state else "fallback",
                },
                direction={
                    "direction": direction_label,
                    "up_probability": up_p,
                    "agreement": dir_agree,
                    "source": "xgb" if xgb_block.get("probability") else "fallback",
                },
                horizon=int(hz.get("horizon_days") or 5),
            )

    exhaustion = float((inputs.get("dealer") or {}).get("dealer_exhaustion") or 0.0)
    risk_rating = _risk_rating(probability, exhaustion, magnitude, confidence)

    # Attach composite into forecast meta for downstream consumers
    fc.setdefault("meta", {})
    fc["meta"]["composite"] = {
        "scores": scores.to_dict(),
        "probability": probability,
        "magnitude": magnitude,
        "expected_duration": expected_duration,
        "expected_start": expected_start,
        "confidence": confidence,
        "risk_rating": risk_rating,
    }
    fc["model_versions"] = {
        **(fc.get("model_versions") or {}),
        "composite": "composite-squeeze-v1",
        "schema": SCHEMA_VERSION,
    }

    return CompositeSqueezeResult(
        symbol=sym,
        as_of=as_of,
        scores=scores,
        probability=probability,
        magnitude=magnitude,
        expected_duration=expected_duration,
        expected_start=expected_start,
        confidence=confidence,
        risk_rating=risk_rating,
        inputs={k: v for k, v in inputs.items() if k != "feature_row"},
        forecast=fc,
        meta={
            "alpaca_configured": alpaca_configured(),
            "input_errors": errors,
            "primary_horizon_days": (h5 or {}).get("horizon_days", 5),
        },
    )
