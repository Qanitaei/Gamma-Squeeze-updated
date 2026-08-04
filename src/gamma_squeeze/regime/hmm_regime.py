"""Hidden Markov Model market-regime classifier with named gamma/vol/trend states."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gamma_squeeze.config import resolve_models_root

try:
    from hmmlearn.hmm import GaussianHMM

    HAS_HMMLEARN = True
except ImportError:  # pragma: no cover
    HAS_HMMLEARN = False
    GaussianHMM = None  # type: ignore


# Canonical mutually exclusive hidden states (order is stable / API contract)
REGIME_STATES: list[str] = [
    "Bull",
    "Strong Bull",
    "Neutral",
    "Compression",
    "Distribution",
    "Accumulation",
    "Positive Gamma",
    "Negative Gamma",
    "Dealer Neutral",
    "High Volatility",
    "Low Volatility",
    "Gamma Expansion",
    "Gamma Squeeze",
    "Short Squeeze",
    "Capitulation",
    "Panic",
]

N_STATES = len(REGIME_STATES)
STATE_INDEX = {name: i for i, name in enumerate(REGIME_STATES)}

# Locked API output fields
HMM_OUTPUT_FIELDS: tuple[str, ...] = (
    "current_state",
    "transition_matrix",
    "next_state_probability",
    "confidence",
)

# Legacy 3-state labels retained for old checkpoints
LEGACY_REGIME_NAMES = {
    0: "low_vol_trend",
    1: "high_vol_stress",
    2: "mean_revert",
}

OBS_FEATURE_CANDIDATES = [
    "ret_1d",
    "ret_5d",
    "rvol_10d",
    "positioning_stress",
    "gex_per_spot",
    "net_gex",
    "regime_neg_gamma",
    "pcr_oi",
    "dealer_gamma",
    "dealer_hedge_requirement",
    "atr",
    "bollinger_width",
    "rsi",
    "adx",
    "vix",
    "spot",
]


@dataclass
class HMMRegimeModel:
    n_states: int = N_STATES
    model: Any = None  # optional GaussianHMM for continuous refinement
    feature_columns: list[str] = field(default_factory=list)
    version: str = "hmm-named-regimes-v2"
    state_labels: list[str] = field(default_factory=lambda: list(REGIME_STATES))
    transition_matrix: np.ndarray | None = None
    emission_templates: np.ndarray | None = None  # (K, F) centroids in obs space
    prior: np.ndarray | None = None

    def infer(self, X: np.ndarray, *, features: pd.DataFrame | None = None) -> dict[str, Any]:
        labels = _normalize_labels(self.state_labels, self.n_states)
        self.state_labels = labels
        if len(X) == 0:
            return _empty_inference(labels)

        # Legacy anonymous GaussianHMM checkpoints (≠ named 16-state pack)
        if self.model is not None and self.n_states != N_STATES and hasattr(self.model, "predict_proba"):
            return _infer_legacy_gaussian(self, X, labels)

        emissions = _emission_probs(X, features=features, columns=self.feature_columns)
        if self.transition_matrix is None or self.transition_matrix.shape[0] != N_STATES:
            hard = emissions.argmax(axis=1)
            self.transition_matrix = _estimate_transitions(hard, N_STATES)
            self.n_states = N_STATES
            labels = list(REGIME_STATES)
            self.state_labels = labels

        # Optional continuous HMM blend (maps anonymous components → named states)
        if self.model is not None and hasattr(self.model, "predict_proba"):
            try:
                gauss_proba = np.asarray(self.model.predict_proba(X), dtype=float)
                mapped = _map_gaussian_to_named(
                    gauss_proba,
                    self.model,
                    self.emission_templates,
                    self.feature_columns,
                )
                emissions = 0.55 * emissions + 0.45 * mapped
                emissions = emissions / np.clip(emissions.sum(axis=1, keepdims=True), 1e-12, None)
            except Exception:  # noqa: BLE001
                pass

        posterior, path = _forward_filter(emissions, self.transition_matrix, self.prior)
        current_id = int(path[-1])
        current_post = posterior[-1]
        next_probs = current_post @ self.transition_matrix
        confidence = _confidence(current_post)

        return {
            "current_state": labels[current_id],
            "regime": labels[current_id],  # backward compatible
            "regime_id": current_id,
            "confidence": confidence,
            "state_posterior": {labels[i]: float(current_post[i]) for i in range(len(labels))},
            "next_state_probability": {labels[i]: float(next_probs[i]) for i in range(len(labels))},
            "transition_matrix": _labeled_transition(self.transition_matrix, labels),
            "transition_matrix_array": np.asarray(self.transition_matrix).tolist(),
            "state_labels": list(labels),
            "proba": [float(x) for x in current_post],
            "path": [labels[int(s)] for s in path[-30:]],
            "path_ids": [int(s) for s in path[-30:]],
        }


def _normalize_labels(labels: list[str] | dict[int, str] | None, n_states: int) -> list[str]:
    if isinstance(labels, dict):
        return [str(labels.get(i, f"state_{i}")) for i in range(n_states)]
    if isinstance(labels, list) and len(labels) == n_states:
        return list(labels)
    if n_states == N_STATES:
        return list(REGIME_STATES)
    return [LEGACY_REGIME_NAMES.get(i, f"state_{i}") for i in range(n_states)]


def _empty_inference(labels: list[str]) -> dict[str, Any]:
    n = len(labels)
    uniform = {labels[i]: 1.0 / n for i in range(n)}
    neutral = "Neutral" if "Neutral" in labels else labels[min(2, n - 1)]
    return {
        "current_state": neutral,
        "regime": "unknown",
        "regime_id": labels.index(neutral) if neutral in labels else 0,
        "confidence": 0.0,
        "state_posterior": uniform,
        "next_state_probability": dict(uniform),
        "transition_matrix": {a: dict(uniform) for a in labels},
        "transition_matrix_array": (np.ones((n, n)) / n).tolist(),
        "state_labels": list(labels),
        "proba": [1.0 / n] * n,
        "path": [],
        "path_ids": [],
    }


def _infer_legacy_gaussian(model: HMMRegimeModel, X: np.ndarray, labels: list[str]) -> dict[str, Any]:
    path = np.asarray(model.model.predict(X), dtype=int)
    proba = np.asarray(model.model.predict_proba(X), dtype=float)
    trans = np.asarray(
        getattr(model.model, "transmat_", model.transition_matrix),
        dtype=float,
    )
    if trans is None:
        trans = _estimate_transitions(path, len(labels))
    current_id = int(path[-1])
    current_post = proba[-1]
    next_probs = current_post @ trans
    return {
        "current_state": labels[current_id],
        "regime": labels[current_id],
        "regime_id": current_id,
        "confidence": _confidence(current_post),
        "state_posterior": {labels[i]: float(current_post[i]) for i in range(len(labels))},
        "next_state_probability": {labels[i]: float(next_probs[i]) for i in range(len(labels))},
        "transition_matrix": _labeled_transition(trans, labels),
        "transition_matrix_array": trans.tolist(),
        "state_labels": list(labels),
        "proba": [float(x) for x in current_post],
        "path": [labels[int(s)] for s in path[-30:]],
        "path_ids": [int(s) for s in path[-30:]],
    }


def _build_obs(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in OBS_FEATURE_CANDIDATES if c in features.columns and c != "spot"]
    if not cols:
        if "spot" in features.columns:
            s = features["spot"].astype(float)
            ret = s.pct_change().fillna(0.0)
            rvol = ret.rolling(10).std().fillna(0.02)
            X = np.column_stack([ret.to_numpy(), rvol.to_numpy()])
            return np.asarray(X, dtype=float), ["ret_1d", "rvol_10d"]
        return np.zeros((0, 1)), []

    frame = features[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = np.array(frame.to_numpy(), dtype=float, copy=True)
    for name in ("gex_per_spot", "net_gex", "dealer_gamma", "dealer_hedge_requirement"):
        if name in cols:
            i = cols.index(name)
            scale = 1e6 if name != "dealer_hedge_requirement" else 1e5
            X[:, i] = np.tanh(X[:, i] / scale)
    if "vix" in cols:
        i = cols.index("vix")
        X[:, i] = np.clip(X[:, i] / 100.0, 0.0, 1.5)
    return X, cols


def _col(features: pd.DataFrame | None, columns: list[str], X: np.ndarray, name: str) -> np.ndarray:
    n = len(X)
    if features is not None and name in features.columns:
        return pd.to_numeric(features[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if name in columns:
        return X[:, columns.index(name)]
    return np.zeros(n, dtype=float)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


def _emission_probs(
    X: np.ndarray,
    *,
    features: pd.DataFrame | None,
    columns: list[str],
) -> np.ndarray:
    """Rule-informed emission probabilities over named regimes (shape T×K)."""
    n = len(X)
    ret1 = _col(features, columns, X, "ret_1d")
    ret5 = _col(features, columns, X, "ret_5d")
    if np.allclose(ret5, 0) and not np.allclose(ret1, 0):
        # synthesize 5d from rolling sum of 1d when missing
        s = pd.Series(ret1).rolling(5, min_periods=1).sum().to_numpy()
        ret5 = s
    rvol = _col(features, columns, X, "rvol_10d")
    if np.allclose(rvol, 0):
        rvol = pd.Series(ret1).rolling(10, min_periods=2).std().fillna(0.15).to_numpy()
    stress = _col(features, columns, X, "positioning_stress")
    gex = _col(features, columns, X, "gex_per_spot")
    if np.allclose(gex, 0):
        gex = _col(features, columns, X, "net_gex")
    neg_g = _col(features, columns, X, "regime_neg_gamma")
    pcr = _col(features, columns, X, "pcr_oi")
    bbw = _col(features, columns, X, "bollinger_width")
    rsi = _col(features, columns, X, "rsi")
    adx = _col(features, columns, X, "adx")
    vix = _col(features, columns, X, "vix")
    atr = _col(features, columns, X, "atr")

    # normalize helpers
    vol_z = (rvol - np.nanmedian(rvol)) / (np.nanstd(rvol) + 1e-6)
    if np.nanstd(vix) > 0:
        vol_z = 0.5 * vol_z + 0.5 * ((vix - np.nanmedian(vix)) / (np.nanstd(vix) + 1e-6))
    trend = 0.6 * ret1 + 0.4 * ret5
    # gex already tanh-scaled when from X; raw frame may not be
    gex_s = np.tanh(np.asarray(gex, dtype=float) / (1.0 if np.nanmax(np.abs(gex)) < 5 else 1e6))

    logits = np.zeros((n, N_STATES), dtype=float)

    def add(name: str, score: np.ndarray) -> None:
        logits[:, STATE_INDEX[name]] += score

    # Trend / risk-off
    add("Bull", 2.2 * np.clip(trend, 0, None) * 40 + 0.4 * (vol_z < 0).astype(float))
    add("Strong Bull", 3.0 * np.clip(trend - 0.01, 0, None) * 50 + 0.8 * (adx / 50.0))
    add("Neutral", 1.5 - 25.0 * np.abs(trend) - 0.4 * np.abs(vol_z))
    add("Capitulation", 3.0 * np.clip(-trend - 0.015, 0, None) * 55 + 1.2 * np.clip(vol_z, 0, None))
    add("Panic", 3.5 * np.clip(-ret1 - 0.025, 0, None) * 60 + 2.0 * np.clip(vol_z - 0.5, 0, None))

    # Structure
    positive_bbw = bbw[bbw > 0]
    bbw_med = float(np.nanmedian(positive_bbw)) if positive_bbw.size else 0.05
    compress = (bbw > 0).astype(float) * (1.0 - np.clip(bbw / (bbw_med + 1e-6), 0, 2))
    if np.allclose(bbw, 0):
        compress = np.clip(0.8 - np.abs(vol_z), 0, None)
    add("Compression", 2.0 * compress + 0.6 * (vol_z < -0.2).astype(float))
    add("Distribution", 2.0 * np.clip(-ret5, 0, None) * 35 + 1.0 * np.clip(pcr - 1.0, 0, None) + 0.5 * (rsi > 60).astype(float) * 0.02 * np.maximum(rsi - 60, 0))
    add("Accumulation", 2.0 * np.clip(ret5, 0, None) * 25 + 1.0 * np.clip(1.0 - pcr, 0, None) + 0.8 * (rsi < 45).astype(float) * (trend > 0).astype(float))

    # Dealer / gamma
    add("Positive Gamma", 2.5 * np.clip(gex_s, 0, None) * 3 + 1.0 * (neg_g < 0.5).astype(float))
    add("Negative Gamma", 2.5 * np.clip(-gex_s, 0, None) * 3 + 1.5 * neg_g + 0.8 * stress)
    add("Dealer Neutral", 1.8 - 2.5 * np.abs(gex_s) - 0.6 * stress)
    add("Gamma Expansion", 2.0 * np.abs(gex_s) * 2 + 1.0 * np.clip(vol_z, 0, None) + 0.5 * stress)
    # Squeeze: neg gamma + upside thrust + elevated vol
    squeeze = (
        2.5 * np.clip(-gex_s, 0, None)
        + 2.5 * np.clip(ret1, 0, None) * 40
        + 1.5 * np.clip(vol_z, 0, None)
        + 1.0 * stress
    )
    add("Gamma Squeeze", squeeze)
    add(
        "Short Squeeze",
        2.0 * np.clip(ret1 - 0.02, 0, None) * 50
        + 1.5 * np.clip(ret5, 0, None) * 30
        + 1.0 * np.clip(vol_z, 0, None)
        + 0.8 * np.clip(pcr - 1.0, 0, None),
    )

    # Pure vol
    add("High Volatility", 2.5 * np.clip(vol_z, 0, None) + 0.8 * (atr > 0).astype(float) * np.clip(vol_z, 0, None))
    add("Low Volatility", 2.5 * np.clip(-vol_z, 0, None) + 1.0 * compress)

    return _softmax(logits, axis=1)


def _estimate_transitions(path: np.ndarray, n_states: int, *, alpha: float = 0.5) -> np.ndarray:
    counts = np.full((n_states, n_states), alpha, dtype=float)
    for a, b in zip(path[:-1], path[1:]):
        counts[int(a), int(b)] += 1.0
    # stickiness prior on diagonal
    counts[np.arange(n_states), np.arange(n_states)] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def _forward_filter(
    emissions: np.ndarray,
    trans: np.ndarray,
    prior: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    t_steps, k = emissions.shape
    post = np.zeros_like(emissions)
    path = np.zeros(t_steps, dtype=int)
    p0 = prior if prior is not None else np.ones(k) / k
    post[0] = p0 * emissions[0]
    post[0] /= max(post[0].sum(), 1e-12)
    path[0] = int(np.argmax(post[0]))
    for t in range(1, t_steps):
        pred = post[t - 1] @ trans
        post[t] = pred * emissions[t]
        post[t] /= max(post[t].sum(), 1e-12)
        path[t] = int(np.argmax(post[t]))
    return post, path


def _confidence(posterior: np.ndarray) -> float:
    """Confidence from top-mass and inverse entropy."""
    p = np.clip(posterior, 1e-12, 1.0)
    p = p / p.sum()
    top = float(p.max())
    ent = float(-(p * np.log(p)).sum())
    max_ent = float(np.log(len(p)))
    ent_score = 1.0 - (ent / max_ent if max_ent > 0 else 0.0)
    return float(np.clip(0.65 * top + 0.35 * ent_score, 0.0, 1.0))


def _labeled_transition(trans: np.ndarray, labels: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, a in enumerate(labels):
        out[a] = {labels[j]: float(trans[i, j]) for j in range(len(labels))}
    return out


def _map_gaussian_to_named(
    gauss_proba: np.ndarray,
    model: Any,
    templates: np.ndarray | None,
    columns: list[str],
) -> np.ndarray:
    """Map anonymous GaussianHMM component probs onto named-state simplex."""
    n_comp = gauss_proba.shape[1]
    mapping = np.zeros((n_comp, N_STATES), dtype=float)
    means = getattr(model, "means_", None)
    if means is None or templates is None or templates.size == 0:
        # uniform dump into Neutral / High-Low vol by component index
        for c in range(n_comp):
            mapping[c, STATE_INDEX["Neutral"]] = 1.0
        return gauss_proba @ mapping

    # templates: (K, F) — align dims
    f = min(means.shape[1], templates.shape[1])
    m = means[:, :f]
    t = templates[:, :f]
    m_n = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    t_n = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    sim = m_n @ t_n.T  # (C, K)
    soft = _softmax(3.0 * sim, axis=1)
    return gauss_proba @ soft


def _state_templates(columns: list[str]) -> np.ndarray:
    """Hand-crafted centroids in observation space for Gaussian mapping."""
    templates = np.zeros((N_STATES, len(columns)), dtype=float)

    def setv(state: str, **kwargs: float) -> None:
        i = STATE_INDEX[state]
        for k, v in kwargs.items():
            if k in columns:
                templates[i, columns.index(k)] = v

    setv("Bull", ret_1d=0.008, ret_5d=0.02, rvol_10d=0.15, gex_per_spot=0.3)
    setv("Strong Bull", ret_1d=0.02, ret_5d=0.05, rvol_10d=0.22, adx=35)
    setv("Neutral", ret_1d=0.0, ret_5d=0.0, rvol_10d=0.14, gex_per_spot=0.0)
    setv("Compression", rvol_10d=0.08, bollinger_width=0.03, ret_1d=0.0)
    setv("Distribution", ret_5d=-0.02, pcr_oi=1.2, rsi=65)
    setv("Accumulation", ret_5d=0.015, pcr_oi=0.8, rsi=40)
    setv("Positive Gamma", gex_per_spot=0.6, regime_neg_gamma=0.0, rvol_10d=0.12)
    setv("Negative Gamma", gex_per_spot=-0.6, regime_neg_gamma=1.0, positioning_stress=0.7)
    setv("Dealer Neutral", gex_per_spot=0.0, positioning_stress=0.2)
    setv("High Volatility", rvol_10d=0.35, vix=0.28)
    setv("Low Volatility", rvol_10d=0.08, vix=0.12)
    setv("Gamma Expansion", gex_per_spot=-0.3, rvol_10d=0.25, positioning_stress=0.6)
    setv("Gamma Squeeze", ret_1d=0.025, gex_per_spot=-0.7, rvol_10d=0.3, positioning_stress=0.8)
    setv("Short Squeeze", ret_1d=0.03, ret_5d=0.06, rvol_10d=0.28, pcr_oi=1.3)
    setv("Capitulation", ret_1d=-0.03, ret_5d=-0.06, rvol_10d=0.32)
    setv("Panic", ret_1d=-0.04, rvol_10d=0.45, vix=0.4)
    return templates


def fit_hmm_regimes(features: pd.DataFrame, *, n_states: int = N_STATES) -> HMMRegimeModel:
    """Fit named-regime HMM (transition matrix + optional GaussianHMM blend)."""
    X, cols = _build_obs(features)
    labels = list(REGIME_STATES) if n_states == N_STATES else [f"state_{i}" for i in range(n_states)]
    bundle = HMMRegimeModel(
        n_states=len(labels),
        feature_columns=cols,
        state_labels=labels,
        emission_templates=_state_templates(cols) if cols else None,
    )
    if len(X) < 8:
        bundle.transition_matrix = np.ones((bundle.n_states, bundle.n_states)) / bundle.n_states
        bundle.prior = np.ones(bundle.n_states) / bundle.n_states
        return bundle

    emissions = _emission_probs(X, features=features, columns=cols)
    # If caller requested fewer anonymous states, fall back to classic GaussianHMM
    if n_states != N_STATES and HAS_HMMLEARN and len(X) >= max(15, n_states * 5):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=42,
        )
        model.fit(X)
        bundle.model = model
        bundle.n_states = n_states
        bundle.state_labels = [LEGACY_REGIME_NAMES.get(i, f"state_{i}") for i in range(n_states)]
        bundle.transition_matrix = np.asarray(model.transmat_, dtype=float)
        bundle.version = "hmm-gaussian-legacy-v1"
        bundle.prior = np.asarray(getattr(model, "startprob_", np.ones(n_states) / n_states), dtype=float)
        return bundle

    hard = emissions.argmax(axis=1)
    bundle.transition_matrix = _estimate_transitions(hard, bundle.n_states)
    bundle.prior = emissions[0] / max(emissions[0].sum(), 1e-12)

    # Optional continuous refinement when data-rich
    if HAS_HMMLEARN and len(X) >= max(48, N_STATES * 3):
        try:
            gmm = GaussianHMM(
                n_components=min(8, max(3, len(X) // 20)),
                covariance_type="diag",
                n_iter=80,
                random_state=42,
            )
            gmm.fit(X)
            bundle.model = gmm
            bundle.version = "hmm-named-regimes-v2+gaussian"
        except Exception:  # noqa: BLE001
            pass

    return bundle


def infer_regime(features: pd.DataFrame, model: HMMRegimeModel | None = None) -> dict[str, Any]:
    m = model or load_hmm_model() or fit_hmm_regimes(features)
    X, cols = _build_obs(features)
    if not m.feature_columns:
        m.feature_columns = cols
    out = m.infer(X, features=features)
    out["version"] = m.version
    out["n_obs"] = int(len(X))
    out["feature_columns"] = list(m.feature_columns)
    return out


def classify_regime(features: pd.DataFrame, *, train: bool = True) -> dict[str, Any]:
    """Fit (optional) + infer named regime block for API consumers."""
    model = fit_hmm_regimes(features) if train else (load_hmm_model() or fit_hmm_regimes(features))
    if train and model.model is not None:
        save_hmm_model(model)
    elif train:
        save_hmm_model(model)
    return infer_regime(features, model)


def save_hmm_model(model: HMMRegimeModel, *, root: Path | None = None) -> Path:
    path = (root or resolve_models_root()) / "regime_hmm" / "latest.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_hmm_model(*, root: Path | None = None) -> HMMRegimeModel | None:
    path = (root or resolve_models_root()) / "regime_hmm" / "latest.joblib"
    if not path.is_file():
        return None
    obj = joblib.load(path)
    # migrate legacy checkpoints
    if isinstance(obj, HMMRegimeModel) and obj.n_states != N_STATES and not obj.state_labels:
        obj.state_labels = [LEGACY_REGIME_NAMES.get(i, f"state_{i}") for i in range(obj.n_states)]
    return obj
