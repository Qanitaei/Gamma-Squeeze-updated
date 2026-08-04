from __future__ import annotations

from typing import Any

from gamma_squeeze.features.feature_store import build_features_for_symbol, load_features
from gamma_squeeze.regime.hmm_regime import (
    REGIME_STATES,
    classify_regime,
    fit_hmm_regimes,
    infer_regime,
    save_hmm_model,
)


def run_regime(
    symbol: str,
    *,
    train: bool = True,
    include_macro: bool = True,
    include_technicals: bool = True,
) -> dict[str, Any]:
    """Classify market regime via 16-state HMM; returns current state, P, next probs, confidence."""
    sym = symbol.upper()
    df = load_features(sym)
    if df.empty:
        df = build_features_for_symbol(
            sym,
            include_skew=False,
            include_macro=include_macro,
            include_technicals=include_technicals,
            include_dealer=True,
            include_options_metrics=True,
        )
    if df.empty:
        n = len(REGIME_STATES)
        uniform = {s: 1.0 / n for s in REGIME_STATES}
        return {
            "symbol": sym,
            "error": "no_features",
            "current_state": "Neutral",
            "transition_matrix": {a: dict(uniform) for a in REGIME_STATES},
            "next_state_probability": dict(uniform),
            "confidence": 0.0,
            "regime": "unknown",
            "state_labels": list(REGIME_STATES),
        }

    if train:
        model = fit_hmm_regimes(df)
        save_hmm_model(model)
        out = infer_regime(df, model)
    else:
        out = classify_regime(df, train=False)

    out["symbol"] = sym
    out["as_of"] = str(df.sort_values("as_of").iloc[-1]["as_of"])
    # Canonical API surface (locked outputs first)
    return {
        "symbol": sym,
        "as_of": out["as_of"],
        "current_state": out.get("current_state") or out.get("regime"),
        "transition_matrix": out.get("transition_matrix"),
        "next_state_probability": out.get("next_state_probability"),
        "confidence": out.get("confidence"),
        "state_posterior": out.get("state_posterior"),
        "state_labels": out.get("state_labels") or list(REGIME_STATES),
        "hidden_states": list(REGIME_STATES),
        "path": out.get("path"),
        "version": out.get("version"),
        "n_obs": out.get("n_obs"),
        "feature_columns": out.get("feature_columns"),
        # extras / backward compatible
        "regime": out.get("regime"),
        "regime_id": out.get("regime_id"),
        "proba": out.get("proba"),
        "transition_matrix_array": out.get("transition_matrix_array"),
    }
