"""CNN / LSTM / Transformer hybrid encoder for OHLCV pattern scoring (numpy)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(np.clip(z, -40, 40))
    return e / max(float(e.sum()), 1e-12)


def _normalize_window(ohlcv: np.ndarray) -> np.ndarray:
    """ohlcv (T, 5) → z-scored channels."""
    x = np.asarray(ohlcv, dtype=float)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0) + 1e-9
    return (x - mu) / sd


def cnn_features(window: np.ndarray) -> np.ndarray:
    """1D multi-kernel conv features over OHLCV (edge / slope / compression)."""
    x = _normalize_window(window)  # (T, C)
    t, c = x.shape
    kernels = [
        np.array([-1.0, 0.0, 1.0]),  # edge
        np.array([-1.0, 2.0, -1.0]),  # curvature
        np.array([1.0, 1.0, 1.0]) / 3.0,  # smooth
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / 3.0,  # trend
    ]
    feats: list[float] = []
    close = x[:, 3] if c > 3 else x[:, 0]
    for k in kernels:
        # convolve close
        if len(close) < len(k):
            feats.extend([0.0, 0.0])
            continue
        conv = np.convolve(close, k, mode="valid")
        feats.append(float(conv[-1]))
        feats.append(float(np.std(conv[-min(5, len(conv)) :])))
    # channel-wise short energy
    for j in range(min(c, 5)):
        feats.append(float(np.std(x[-5:, j])))
        feats.append(float(x[-1, j] - x[0, j]))
    return np.asarray(feats, dtype=float)


def lstm_features(window: np.ndarray, hidden: int = 8) -> np.ndarray:
    """Lightweight GRU-like recurrent readout (fixed random orthogonal gates)."""
    x = _normalize_window(window)
    rng = np.random.default_rng(42)
    din = x.shape[1]
    # fixed pseudo-learned weights (deterministic)
    wz = rng.normal(0, 0.3, size=(hidden, din))
    uz = rng.normal(0, 0.3, size=(hidden, hidden))
    wr = rng.normal(0, 0.3, size=(hidden, din))
    ur = rng.normal(0, 0.3, size=(hidden, hidden))
    wh = rng.normal(0, 0.3, size=(hidden, din))
    uh = rng.normal(0, 0.3, size=(hidden, hidden))

    h = np.zeros(hidden, dtype=float)
    for t in range(x.shape[0]):
        xt = x[t]
        z = 1.0 / (1.0 + np.exp(-np.clip(wz @ xt + uz @ h, -40, 40)))
        r = 1.0 / (1.0 + np.exp(-np.clip(wr @ xt + ur @ h, -40, 40)))
        h_tilde = np.tanh(wh @ xt + uh @ (r * h))
        h = (1 - z) * h + z * h_tilde
    return h


def transformer_features(window: np.ndarray, d_model: int = 8, n_heads: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Multi-head self-attention pooling; returns context + mean attention weights over time."""
    x = _normalize_window(window)
    t, c = x.shape
    rng = np.random.default_rng(7)
    # project bars into d_model
    w_in = rng.normal(0, 0.4, size=(c, d_model))
    e = x @ w_in  # (T, d)
    # positional encoding
    pos = np.arange(t)[:, None]
    dims = np.arange(d_model)[None, :]
    pe = np.sin(pos / (10000 ** (dims / max(d_model, 1))))
    e = e + 0.1 * pe

    head_dim = max(d_model // n_heads, 1)
    contexts = []
    attn_acc = np.zeros(t, dtype=float)
    for h in range(n_heads):
        wq = rng.normal(0, 0.3, size=(d_model, head_dim))
        wk = rng.normal(0, 0.3, size=(d_model, head_dim))
        wv = rng.normal(0, 0.3, size=(d_model, head_dim))
        q = e @ wq
        k = e @ wk
        v = e @ wv
        scores = (q @ k.T) / np.sqrt(head_dim)
        # attend from last token
        a = _softmax(scores[-1])
        attn_acc += a
        contexts.append(a @ v)
    ctx = np.concatenate(contexts) if contexts else np.zeros(d_model)
    attn = attn_acc / max(n_heads, 1)
    return ctx.astype(float), attn.astype(float)


def hybrid_embedding(window: np.ndarray) -> dict[str, Any]:
    """Fuse CNN + LSTM + Transformer branches into one embedding + attentions."""
    cnn = cnn_features(window)
    lstm = lstm_features(window)
    trans, attn = transformer_features(window)
    # pad/trim to stable concat
    emb = np.concatenate([cnn, lstm, trans])
    return {
        "embedding": emb,
        "cnn": cnn,
        "lstm": lstm,
        "transformer": trans,
        "temporal_attention": attn,
        "branch_norms": {
            "cnn": float(np.linalg.norm(cnn)),
            "lstm": float(np.linalg.norm(lstm)),
            "transformer": float(np.linalg.norm(trans)),
        },
    }


# Pattern prototypes in a reduced score space (hand-crafted directions)
_PATTERN_DIRECTIONS: dict[str, np.ndarray] = {}


def _stable_seed(name: str) -> int:
    # deterministic across processes (avoid PYTHONHASHSEED variance)
    h = 2166136261
    for ch in name.encode("utf-8"):
        h = (h ^ ch) * 16777619
        h &= 0xFFFFFFFF
    return int(h)


def _prototype(name: str, dim: int) -> np.ndarray:
    if name not in _PATTERN_DIRECTIONS or len(_PATTERN_DIRECTIONS[name]) != dim:
        rng = np.random.default_rng(_stable_seed(name))
        v = rng.normal(0, 1, size=dim)
        _PATTERN_DIRECTIONS[name] = v / (np.linalg.norm(v) + 1e-9)
    return _PATTERN_DIRECTIONS[name]


def hybrid_pattern_scores(window: np.ndarray, pattern_names: list[str]) -> dict[str, float]:
    """Cosine similarity of hybrid embedding to pattern prototypes → [0,1] scores."""
    emb = hybrid_embedding(window)["embedding"]
    n = np.linalg.norm(emb) + 1e-9
    emb_n = emb / n
    out: dict[str, float] = {}
    for name in pattern_names:
        proto = _prototype(name, len(emb_n))
        sim = float(np.dot(emb_n, proto))
        out[name] = float(np.clip(0.5 * (sim + 1.0), 0.0, 1.0))
    return out
