"""Ensemble inference assembling full SqueezeForecast payloads."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gamma_squeeze.config import HORIZONS
from gamma_squeeze.dealer.hedge_demand import compute_hedge_demand_curve
from gamma_squeeze.explain.engine import explain_prediction
from gamma_squeeze.models.distribution import DistributionModelBundle, fit_distribution_residuals
from gamma_squeeze.models.duration import DurationModelBundle, load_duration_bundle
from gamma_squeeze.models.magnitude import MagnitudeModelBundle, load_magnitude_bundle
from gamma_squeeze.models.squeeze_probability import ProbabilityModelBundle, load_probability_bundle
from gamma_squeeze.risk.portfolio_hedges import recommend_portfolio_hedges
from gamma_squeeze.risk.trade_recommendations import recommend_trades
from gamma_squeeze.serve.schemas import (
    SCHEMA_VERSION,
    DistributionBin,
    HorizonForecast,
    MagnitudeQuantiles,
    SqueezeForecast,
)

# Regimes that agree with a long-gamma / squeeze thesis
_SQUEEZE_REGIMES = frozenset(
    {
        "Gamma Squeeze",
        "Short Squeeze",
        "Negative Gamma",
        "Gamma Expansion",
        "Panic",
        "High Volatility",
        "Capitulation",
    }
)


class SqueezeEnsemble:
    def __init__(
        self,
        probability: ProbabilityModelBundle | None = None,
        magnitude: MagnitudeModelBundle | None = None,
        duration: DurationModelBundle | None = None,
        distribution: DistributionModelBundle | None = None,
    ):
        self.probability = probability or load_probability_bundle() or ProbabilityModelBundle()
        self.magnitude = magnitude or load_magnitude_bundle() or MagnitudeModelBundle()
        self.duration = duration or load_duration_bundle() or DurationModelBundle()
        self.distribution = distribution or DistributionModelBundle()

    @classmethod
    def load_latest(cls) -> "SqueezeEnsemble":
        return cls()

    def predict(
        self,
        feature_row: pd.Series | dict[str, Any],
        *,
        symbol: str,
        as_of: str,
        matrix: dict[str, Any] | None = None,
        feature_panel: pd.DataFrame | None = None,
    ) -> SqueezeForecast:
        row = feature_row.to_dict() if isinstance(feature_row, pd.Series) else dict(feature_row)
        row.setdefault("symbol", symbol.upper())

        probs = self.probability.predict_proba_row(row) if self.probability.models else {h: 0.0 for h in HORIZONS}
        mags = self.magnitude.predict_row(row) if self.magnitude.models_p50 else {
            h: {"p10": 0.0, "p50": 0.0, "p90": 0.0, "expected": 0.0} for h in HORIZONS
        }
        durs = self.duration.predict_row(row)

        regime_sig = _regime_signal(row, feature_panel=feature_panel)
        direction_sig = _direction_signal(row)

        horizons: list[HorizonForecast] = []
        for h in HORIZONS:
            mq = mags.get(h, {"p10": 0.0, "p50": 0.0, "p90": 0.0, "expected": 0.0})
            p = float(probs.get(h, 0.0))
            p = max(0.0, min(1.0, p))
            conf = blend_confidence(p, row, regime=regime_sig, direction=direction_sig, horizon=h)
            bins = self.distribution.predict_bins(
                horizon=h,
                p10=float(mq["p10"]),
                p50=float(mq["p50"]),
                p90=float(mq["p90"]),
            )
            if not bins:
                bins = [
                    DistributionBin(lo_pct=-5.0, hi_pct=0.0, probability=0.5),
                    DistributionBin(lo_pct=0.0, hi_pct=5.0, probability=0.5),
                ]
            horizons.append(
                HorizonForecast(
                    horizon_days=h,
                    squeeze_probability=p,
                    expected_magnitude_pct=float(mq["expected"]) * 100.0,
                    magnitude_quantiles=MagnitudeQuantiles(
                        p10=float(mq["p10"]) * 100.0,
                        p50=float(mq["p50"]) * 100.0,
                        p90=float(mq["p90"]) * 100.0,
                    ),
                    expected_duration_days=float(durs.get(h, h)),
                    return_distribution=bins,
                    confidence_score=conf,
                )
            )

        hedge = compute_hedge_demand_curve(matrix) if matrix else []
        trades = recommend_trades(horizons, row) or []
        hedges = recommend_portfolio_hedges(horizons, row) or []

        # Every prediction must explain WHY
        expl = explain_prediction(
            symbol,
            row,
            horizons=horizons,
            probability_bundle=self.probability,
            as_of=as_of,
        )
        explanations = list(expl.schema_explanations or [])
        if not explanations:
            from gamma_squeeze.explain.attributions import explain_row

            explanations = explain_row(row, horizons)

        return SqueezeForecast(
            schema_version=SCHEMA_VERSION,
            symbol=symbol.upper(),
            as_of=as_of,
            horizons=horizons,
            dealer_hedging_demand=hedge,
            trade_recommendations=trades,
            portfolio_hedge_recommendations=hedges,
            explanations=explanations,
            model_versions={
                "probability": self.probability.version,
                "magnitude": self.magnitude.version,
                "duration": self.duration.version,
                "distribution": self.distribution.version,
                "explainability": expl.version,
                "hmm_regime": str(regime_sig.get("version") or "unavailable"),
                "xgboost_direction": str(direction_sig.get("version") or "unavailable"),
            },
            meta={
                "positioning_stress": float(row.get("positioning_stress") or 0.0),
                "why": expl.why,
                "explainability": expl.to_dict(),
                "regime": {
                    "state": regime_sig.get("state"),
                    "confidence": regime_sig.get("confidence"),
                    "source": regime_sig.get("source"),
                    "agreement": regime_sig.get("agreement"),
                },
                "direction": {
                    "label": direction_sig.get("direction"),
                    "up_probability": direction_sig.get("up_probability"),
                    "source": direction_sig.get("source"),
                    "agreement": direction_sig.get("agreement"),
                },
                "confidence_blend": {
                    "regime_weight": 0.20,
                    "direction_weight": 0.20,
                    "prob_weight": 0.45,
                    "stress_weight": 0.15,
                },
            },
        )


def blend_confidence(
    squeeze_probability: float,
    row: dict[str, Any],
    *,
    regime: dict[str, Any] | None = None,
    direction: dict[str, Any] | None = None,
    horizon: int = 5,
) -> float:
    """Blend calibrated squeeze prob with regime/direction agreement into [0, 1].

    Weights (sum≈1):
      0.45 calibrated squeeze probability
      0.15 positioning-stress agreement with probability
      0.20 HMM regime agreement (squeeze-friendly × regime confidence)
      0.20 XGBoost direction agreement (up-prob when squeeze thesis is live)
    Missing artifacts degrade gracefully toward prob/stress-only blend.
    """
    del horizon  # reserved for horizon-specific direction heads
    p = float(max(0.0, min(1.0, squeeze_probability)))
    stress = float(row.get("positioning_stress") or 0.0)
    stress = max(0.0, min(1.0, stress))
    stress_agree = 1.0 - abs(p - stress)

    reg = regime or {}
    direc = direction or {}

    regime_agree = float(reg.get("agreement") if reg.get("agreement") is not None else 0.5)
    regime_agree = max(0.0, min(1.0, regime_agree))
    direction_agree = float(direc.get("agreement") if direc.get("agreement") is not None else 0.5)
    direction_agree = max(0.0, min(1.0, direction_agree))

    # Down-weight missing artifact channels toward neutral 0.5 already baked into agreement
    has_regime = str(reg.get("source") or "") not in ("", "fallback", "unavailable")
    has_direction = str(direc.get("source") or "") not in ("", "fallback", "unavailable")
    w_prob, w_stress = 0.45, 0.15
    w_reg = 0.20 if has_regime else 0.10
    w_dir = 0.20 if has_direction else 0.10
    # Redistribute unused mass to probability
    unused = (0.20 - w_reg) + (0.20 - w_dir)
    w_prob += unused
    total = w_prob + w_stress + w_reg + w_dir
    score = (
        w_prob * p
        + w_stress * stress_agree
        + w_reg * regime_agree
        + w_dir * direction_agree
    ) / total
    return float(max(0.0, min(1.0, score)))


def _regime_signal(
    row: dict[str, Any],
    *,
    feature_panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """HMM regime agreement with squeeze thesis; never raises."""
    try:
        from gamma_squeeze.regime.hmm_regime import infer_regime, load_hmm_model

        model = load_hmm_model()
        if model is not None:
            if feature_panel is not None and not feature_panel.empty:
                panel = feature_panel
            else:
                panel = pd.DataFrame([row])
            out = infer_regime(panel, model)
            state = str(out.get("current_state") or out.get("regime") or "")
            conf = float(out.get("confidence") or 0.5)
            squeeze_friendly = 1.0 if state in _SQUEEZE_REGIMES else (
                0.65 if "Gamma" in state or "Volatility" in state else 0.35
            )
            agreement = max(0.0, min(1.0, squeeze_friendly * (0.45 + 0.55 * conf)))
            return {
                "state": state,
                "confidence": conf,
                "agreement": agreement,
                "source": "hmm_artifact",
                "version": out.get("version") or getattr(model, "version", "hmm"),
            }
    except Exception:  # noqa: BLE001
        pass

    # Feature heuristic fallback (no artifact / infer failure)
    neg = float(row.get("regime_neg_gamma") or 0.0)
    stress = float(row.get("positioning_stress") or 0.0)
    agreement = max(0.0, min(1.0, 0.55 * neg + 0.45 * stress))
    return {
        "state": "Negative Gamma" if neg >= 0.5 else "Neutral",
        "confidence": 0.35,
        "agreement": agreement,
        "source": "fallback",
        "version": "heuristic-v1",
    }


def _direction_signal(row: dict[str, Any]) -> dict[str, Any]:
    """XGBoost direction agreement with squeeze (upside) thesis; never raises."""
    try:
        from gamma_squeeze.models.xgboost_direction import load_direction_model

        bundle = load_direction_model()
        if bundle is not None:
            pred = bundle.predict_row(row, horizon=5)
            proba = pred.get("probability") or pred.get("proba") or {}
            up_p = float(proba.get("up") or 0.33)
            direction = str(pred.get("direction") or pred.get("classification") or "flat")
            # Agreement with long-squeeze (upside) thesis tracks P(up).
            if direction == "up":
                agreement = max(up_p, float(pred.get("confidence") or up_p))
            elif direction == "down":
                agreement = up_p
            else:
                agreement = 0.35 + 0.3 * up_p
            return {
                "direction": direction,
                "up_probability": up_p,
                "agreement": max(0.0, min(1.0, agreement)),
                "source": "xgb_artifact",
                "version": pred.get("version") or getattr(bundle, "version", "xgb"),
            }
    except Exception:  # noqa: BLE001
        pass

    ret5 = float(row.get("ret_5d") or 0.0)
    # Map short-horizon momentum to soft up-agreement
    up_p = max(0.0, min(1.0, 0.5 + ret5 * 8.0))
    return {
        "direction": "up" if up_p >= 0.55 else ("down" if up_p <= 0.45 else "flat"),
        "up_probability": up_p,
        "agreement": up_p,
        "source": "fallback",
        "version": "momentum-heuristic-v1",
    }


# Backward-compatible private alias used by older call sites / tests
def _confidence(prob: float, row: dict[str, Any]) -> float:
    return blend_confidence(prob, row, regime=_regime_signal(row), direction=_direction_signal(row))


def train_ensemble_components(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
) -> SqueezeEnsemble:
    from gamma_squeeze.models.duration import save_duration_bundle, train_duration_models
    from gamma_squeeze.models.magnitude import save_magnitude_bundle, train_magnitude_models
    from gamma_squeeze.models.squeeze_probability import (
        save_probability_bundle,
        train_probability_models,
    )

    prob = train_probability_models(features, labels, feature_columns)
    mag = train_magnitude_models(features, labels, feature_columns)
    dur = train_duration_models(features, labels, feature_columns)
    dist = fit_distribution_residuals(labels)
    save_probability_bundle(prob)
    save_magnitude_bundle(mag)
    save_duration_bundle(dur)
    return SqueezeEnsemble(prob, mag, dur, dist)
