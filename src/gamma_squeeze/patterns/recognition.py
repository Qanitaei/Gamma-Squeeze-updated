"""Technical pattern recognition — CNN/LSTM/Transformer hybrid + classic geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gamma_squeeze.patterns.hybrid_encoder import hybrid_embedding, hybrid_pattern_scores

CHART_PATTERNS = [
    "Bull Flag",
    "Bear Flag",
    "Cup Handle",
    "Ascending Triangle",
    "Descending Triangle",
    "Pennant",
    "Double Bottom",
    "Double Top",
    "Head Shoulders",
    "Inverse Head Shoulders",
    "Rectangle",
    "Wedge",
    "Channel",
    "Gap",
    "Island Reversal",
    "Volatility Squeeze",
    "NR7",
    "Inside Bar",
]

CANDLESTICK_PATTERNS = [
    "Doji",
    "Hammer",
    "Shooting Star",
    "Bullish Engulfing",
    "Bearish Engulfing",
    "Morning Star",
    "Evening Star",
    "Three White Soldiers",
    "Three Black Crows",
    "Harami",
    "Piercing Line",
    "Dark Cloud Cover",
]

PATTERN_OUTPUTS = (
    "pattern",
    "confidence",
    "target_projection",
    "probability",
)

PATTERN_BACKEND = "hybrid-cnn-lstm-transformer"

# Legacy names retained for older clients/tests
LEGACY_ALIASES = {
    "range_breakout": "Rectangle",
    "volatility_compression": "Volatility Squeeze",
}


def detect_patterns(
    ohlcv: pd.DataFrame,
    *,
    lookback: int = 80,
    min_confidence: float = 0.40,
    include_candles: bool = True,
) -> dict[str, Any]:
    """Detect chart + candlestick patterns with hybrid neural scoring."""
    empty = {
        "patterns": [],
        "summary": {"n_bars": 0},
        "backend": PATTERN_BACKEND,
        "outputs": list(PATTERN_OUTPUTS),
        "catalog": {"chart": list(CHART_PATTERNS), "candlestick": list(CANDLESTICK_PATTERNS)},
    }
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        return empty

    df = _ensure_ohlcv(ohlcv.tail(max(lookback, 30)).copy())
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    n = len(close)
    if n < 8:
        empty["summary"] = {"n_bars": n}
        return empty

    window = np.column_stack([open_, high, low, close, volume])
    hybrid = hybrid_embedding(window)
    hybrid_scores = hybrid_pattern_scores(window, CHART_PATTERNS + CANDLESTICK_PATTERNS)

    rule_hits: list[dict[str, Any]] = []
    rule_hits.extend(_detect_chart_patterns(open_, high, low, close, volume))
    if include_candles:
        rule_hits.extend(_detect_candlesticks(open_, high, low, close))

    # Legacy breakout / compression for compatibility
    rule_hits.extend(_legacy_structures(high, low, close, volume))

    fused = _fuse_detections(rule_hits, hybrid_scores, close, high, low, min_confidence=min_confidence)
    fused.sort(key=lambda p: -float(p.get("confidence") or 0))

    bullish = sum(1 for p in fused if (p.get("target_projection") or {}).get("direction") == "up")
    bearish = sum(1 for p in fused if (p.get("target_projection") or {}).get("direction") == "down")
    return {
        "patterns": fused,
        "summary": {
            "n_bars": int(n),
            "n_patterns": len(fused),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "compression": any(p["pattern"] == "Volatility Squeeze" for p in fused),
            "last_close": float(close[-1]),
            "hybrid_branch_norms": hybrid.get("branch_norms"),
        },
        "attention_weights": {
            "temporal": [
                {"lag": int(n - 1 - i), "weight": float(w)}
                for i, w in enumerate(np.asarray(hybrid.get("temporal_attention", []), dtype=float).reshape(-1))
            ]
        },
        "backend": PATTERN_BACKEND,
        "outputs": list(PATTERN_OUTPUTS),
        "catalog": {"chart": list(CHART_PATTERNS), "candlestick": list(CANDLESTICK_PATTERNS)},
    }


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            if col == "volume":
                out[col] = 1.0
            elif col in ("high", "low", "open"):
                out[col] = out["close"]
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["close"])


def _fuse_detections(
    rule_hits: list[dict[str, Any]],
    hybrid_scores: dict[str, float],
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    *,
    min_confidence: float,
) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for hit in rule_hits:
        name = hit["pattern"]
        h_score = hybrid_scores.get(name, 0.45)
        conf = 0.65 * float(hit.get("rule_confidence", 0.5)) + 0.35 * h_score
        prob = float(np.clip(0.55 * conf + 0.25 * h_score + 0.1, 0.05, 0.95))
        target = hit.get("target_projection") or _default_target(name, close, high, low, conf)
        rec = {
            "pattern": name,
            "name": name,  # backward compatible
            "confidence": float(np.clip(conf, 0, 1)),
            "probability": prob,
            "target_projection": target,
            "hybrid_score": h_score,
            "rule_confidence": float(hit.get("rule_confidence", 0.5)),
            "meta": hit.get("meta") or {},
        }
        prev = by_name.get(name)
        if prev is None or rec["confidence"] > prev["confidence"]:
            by_name[name] = rec

    # Promote strong hybrid-only hints (no rule) at lower weight
    for name, h_score in hybrid_scores.items():
        if name in by_name or h_score < 0.78:
            continue
        conf = 0.45 * h_score
        if conf < min_confidence:
            continue
        by_name[name] = {
            "pattern": name,
            "name": name,
            "confidence": float(conf),
            "probability": float(np.clip(0.4 * h_score, 0.05, 0.8)),
            "target_projection": _default_target(name, close, high, low, conf),
            "hybrid_score": h_score,
            "rule_confidence": 0.0,
            "meta": {"source": "hybrid_only"},
        }

    out = [p for p in by_name.values() if p["confidence"] >= min_confidence]
    return out


def _default_target(
    name: str,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    confidence: float,
) -> dict[str, Any]:
    px = float(close[-1])
    rng = float(high[-20:].max() - low[-20:].min()) if len(close) >= 20 else float(high.max() - low.min())
    bullish_names = {
        "Bull Flag",
        "Cup Handle",
        "Ascending Triangle",
        "Double Bottom",
        "Inverse Head Shoulders",
        "Pennant",
        "Channel",
        "Hammer",
        "Bullish Engulfing",
        "Morning Star",
        "Three White Soldiers",
        "Piercing Line",
        "range_breakout",
        "uptrend_structure",
    }
    bearish_names = {
        "Bear Flag",
        "Descending Triangle",
        "Double Top",
        "Head Shoulders",
        "Shooting Star",
        "Bearish Engulfing",
        "Evening Star",
        "Three Black Crows",
        "Dark Cloud Cover",
        "Island Reversal",
    }
    if name in bullish_names:
        direction = "up"
        target = px + rng * (0.5 + 0.5 * confidence)
    elif name in bearish_names:
        direction = "down"
        target = px - rng * (0.5 + 0.5 * confidence)
    else:
        direction = "neutral"
        target = px
    return {
        "price": float(target),
        "pct": float((target / px - 1.0) if px else 0.0),
        "direction": direction,
        "range_used": rng,
    }


# ---------------------------------------------------------------------------
# Chart pattern detectors
# ---------------------------------------------------------------------------


def _detect_chart_patterns(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pivots = _swing_pivots(high, low, close, order=2)
    n = len(close)

    # Flags / pennant after impulse
    out.extend(_flag_pennant(close, high, low, volume))
    # Triangles
    out.extend(_triangles(high, low, close))
    # Double top/bottom
    out.extend(_double_top_bottom(pivots, close, high, low))
    # Head & shoulders
    out.extend(_head_shoulders(pivots, close))
    # Cup & handle
    out.extend(_cup_handle(close, high, low))
    # Rectangle / channel / wedge
    out.extend(_rectangle_channel_wedge(high, low, close))
    # Gaps / island
    out.extend(_gaps_island(open_, high, low, close))
    # Vol squeeze / NR7 / inside bar
    out.extend(_compression_patterns(high, low, close))
    return out


def _flag_pennant(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(close) < 20:
        return out
    move = (close[-8] - close[-20]) / max(abs(close[-20]), 1e-9)
    cons_high = float(high[-7:].max())
    cons_low = float(low[-7:].min())
    cons_width = (cons_high - cons_low) / max(abs(close[-1]), 1e-9)
    vol_shrink = float(np.mean(volume[-5:])) < float(np.mean(volume[-20:-5])) * 0.9
    if move > 0.04 and cons_width < 0.035 and vol_shrink:
        target = float(close[-1] + (close[-8] - close[-20]))
        out.append(
            {
                "pattern": "Bull Flag",
                "rule_confidence": 0.72,
                "target_projection": {
                    "price": target,
                    "pct": target / close[-1] - 1.0,
                    "direction": "up",
                },
                "meta": {"impulse_pct": float(move), "consolidation_width": cons_width},
            }
        )
        # tight consolidation → pennant
        if cons_width < 0.02:
            out.append(
                {
                    "pattern": "Pennant",
                    "rule_confidence": 0.68,
                    "target_projection": {
                        "price": target,
                        "pct": target / close[-1] - 1.0,
                        "direction": "up",
                    },
                    "meta": {"bias": "bullish"},
                }
            )
    if move < -0.04 and cons_width < 0.035 and vol_shrink:
        target = float(close[-1] + (close[-8] - close[-20]))
        out.append(
            {
                "pattern": "Bear Flag",
                "rule_confidence": 0.72,
                "target_projection": {
                    "price": target,
                    "pct": target / close[-1] - 1.0,
                    "direction": "down",
                },
                "meta": {"impulse_pct": float(move), "consolidation_width": cons_width},
            }
        )
    return out


def _triangles(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(close) < 25:
        return out
    # split into thirds for slope of highs/lows
    idx = np.arange(len(close[-24:]))
    h = high[-24:]
    l = low[-24:]
    slope_h = float(np.polyfit(idx, h, 1)[0])
    slope_l = float(np.polyfit(idx, l, 1)[0])
    flat_h = abs(slope_h) < np.std(h) * 0.05
    flat_l = abs(slope_l) < np.std(l) * 0.05
    rising_l = slope_l > 0
    falling_h = slope_h < 0
    px = float(close[-1])
    if flat_h and rising_l:
        target = float(h.max())
        out.append(
            {
                "pattern": "Ascending Triangle",
                "rule_confidence": 0.7,
                "target_projection": {"price": target, "pct": target / px - 1.0, "direction": "up"},
                "meta": {"resistance": float(h.max()), "slope_low": slope_l},
            }
        )
    if flat_l and falling_h:
        target = float(l.min())
        out.append(
            {
                "pattern": "Descending Triangle",
                "rule_confidence": 0.7,
                "target_projection": {"price": target, "pct": target / px - 1.0, "direction": "down"},
                "meta": {"support": float(l.min()), "slope_high": slope_h},
            }
        )
    return out


def _double_top_bottom(
    pivots: list[dict], close: np.ndarray, high: np.ndarray, low: np.ndarray
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    highs = [p for p in pivots if p["type"] == "high"]
    lows = [p for p in pivots if p["type"] == "low"]
    px = float(close[-1])
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if abs(a["price"] - b["price"]) / max(a["price"], 1e-9) < 0.015 and b["i"] - a["i"] >= 3:
            neck = float(low[a["i"] : b["i"] + 1].min())
            height = (a["price"] + b["price"]) / 2 - neck
            target = neck - height
            out.append(
                {
                    "pattern": "Double Top",
                    "rule_confidence": 0.74,
                    "target_projection": {"price": float(target), "pct": target / px - 1.0, "direction": "down"},
                    "meta": {"peaks": [a["price"], b["price"]], "neckline": neck},
                }
            )
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if abs(a["price"] - b["price"]) / max(a["price"], 1e-9) < 0.015 and b["i"] - a["i"] >= 3:
            neck = float(high[a["i"] : b["i"] + 1].max())
            height = neck - (a["price"] + b["price"]) / 2
            target = neck + height
            out.append(
                {
                    "pattern": "Double Bottom",
                    "rule_confidence": 0.74,
                    "target_projection": {"price": float(target), "pct": target / px - 1.0, "direction": "up"},
                    "meta": {"troughs": [a["price"], b["price"]], "neckline": neck},
                }
            )
    return out


def _head_shoulders(pivots: list[dict], close: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    highs = [p for p in pivots if p["type"] == "high"]
    lows = [p for p in pivots if p["type"] == "low"]
    px = float(close[-1])
    if len(highs) >= 3:
        l, h, r = highs[-3], highs[-2], highs[-1]
        if h["price"] > l["price"] and h["price"] > r["price"]:
            if abs(l["price"] - r["price"]) / max(h["price"], 1e-9) < 0.03:
                neck = min(l["price"], r["price"]) * 0.99
                target = neck - (h["price"] - neck)
                out.append(
                    {
                        "pattern": "Head Shoulders",
                        "rule_confidence": 0.73,
                        "target_projection": {
                            "price": float(target),
                            "pct": target / px - 1.0,
                            "direction": "down",
                        },
                        "meta": {"left": l["price"], "head": h["price"], "right": r["price"]},
                    }
                )
    if len(lows) >= 3:
        l, h, r = lows[-3], lows[-2], lows[-1]
        if h["price"] < l["price"] and h["price"] < r["price"]:
            if abs(l["price"] - r["price"]) / max(abs(h["price"]), 1e-9) < 0.03:
                neck = max(l["price"], r["price"]) * 1.01
                target = neck + (neck - h["price"])
                out.append(
                    {
                        "pattern": "Inverse Head Shoulders",
                        "rule_confidence": 0.73,
                        "target_projection": {
                            "price": float(target),
                            "pct": target / px - 1.0,
                            "direction": "up",
                        },
                        "meta": {"left": l["price"], "head": h["price"], "right": r["price"]},
                    }
                )
    return out


def _cup_handle(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(close) < 30:
        return out
    seg = close[-30:]
    left = float(seg[:8].max())
    bottom = float(seg[8:20].min())
    right = float(seg[20:26].max())
    handle_low = float(seg[26:].min())
    depth = (left - bottom) / max(left, 1e-9)
    recovery = abs(right - left) / max(left, 1e-9)
    handle_retr = (right - handle_low) / max(right - bottom, 1e-9)
    if 0.08 < depth < 0.35 and recovery < 0.04 and 0.1 < handle_retr < 0.55:
        px = float(close[-1])
        target = right + (right - bottom)
        out.append(
            {
                "pattern": "Cup Handle",
                "rule_confidence": 0.7,
                "target_projection": {"price": float(target), "pct": target / px - 1.0, "direction": "up"},
                "meta": {"cup_depth": depth, "handle_retracement": handle_retr},
            }
        )
    return out


def _rectangle_channel_wedge(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(close) < 20:
        return out
    idx = np.arange(20)
    h = high[-20:]
    l = low[-20:]
    sh = float(np.polyfit(idx, h, 1)[0])
    sl = float(np.polyfit(idx, l, 1)[0])
    width = float(np.mean(h - l) / max(np.mean(close[-20:]), 1e-9))
    px = float(close[-1])
    # Rectangle: flat bounds
    if abs(sh) < np.std(h) * 0.04 and abs(sl) < np.std(l) * 0.04 and width < 0.08:
        out.append(
            {
                "pattern": "Rectangle",
                "rule_confidence": 0.66,
                "target_projection": {
                    "price": float(h.max()) if px > np.mean(close[-20:]) else float(l.min()),
                    "pct": 0.0,
                    "direction": "up" if px > np.mean(close[-20:]) else "down",
                },
                "meta": {"high": float(h.max()), "low": float(l.min())},
            }
        )
    # Channel: parallel slopes
    if abs(sh - sl) < np.std(h) * 0.05 and abs(sh) > np.std(h) * 0.02:
        direction = "up" if sh > 0 else "down"
        target = px + (h.max() - l.min()) * (1 if direction == "up" else -1)
        out.append(
            {
                "pattern": "Channel",
                "rule_confidence": 0.67,
                "target_projection": {"price": float(target), "pct": target / px - 1.0, "direction": direction},
                "meta": {"slope_high": sh, "slope_low": sl},
            }
        )
    # Wedge: converging
    if sh * sl > 0 and abs(sh) > abs(sl) * 0.2:
        # both rising or both falling with different steepness — converging if width shrinks
        early_w = float(np.mean(h[:7] - l[:7]))
        late_w = float(np.mean(h[-7:] - l[-7:]))
        if late_w < early_w * 0.75:
            direction = "down" if sh > 0 else "up"  # rising wedge bearish, falling wedge bullish
            target = px * (0.97 if direction == "down" else 1.03)
            out.append(
                {
                    "pattern": "Wedge",
                    "rule_confidence": 0.64,
                    "target_projection": {"price": float(target), "pct": target / px - 1.0, "direction": direction},
                    "meta": {"type": "rising" if sh > 0 else "falling"},
                }
            )
    return out


def _gaps_island(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(close) < 5:
        return out
    px = float(close[-1])
    # gap up / down vs prior day
    if low[-1] > high[-2] * 1.001:
        out.append(
            {
                "pattern": "Gap",
                "rule_confidence": 0.7,
                "target_projection": {"price": px * 1.01, "pct": 0.01, "direction": "up"},
                "meta": {"gap_type": "up", "gap_pct": float(low[-1] / high[-2] - 1.0)},
            }
        )
    if high[-1] < low[-2] * 0.999:
        out.append(
            {
                "pattern": "Gap",
                "rule_confidence": 0.7,
                "target_projection": {"price": px * 0.99, "pct": -0.01, "direction": "down"},
                "meta": {"gap_type": "down", "gap_pct": float(high[-1] / low[-2] - 1.0)},
            }
        )
    # Island: gap away then gap back within short window
    if len(close) >= 6:
        gap_away = low[-5] > high[-6]
        gap_back = high[-1] < low[-2]
        if gap_away and gap_back:
            out.append(
                {
                    "pattern": "Island Reversal",
                    "rule_confidence": 0.68,
                    "target_projection": {"price": px * 0.98, "pct": -0.02, "direction": "down"},
                    "meta": {"side": "bearish_island"},
                }
            )
        gap_away_dn = high[-5] < low[-6]
        gap_back_up = low[-1] > high[-2]
        if gap_away_dn and gap_back_up:
            out.append(
                {
                    "pattern": "Island Reversal",
                    "rule_confidence": 0.68,
                    "target_projection": {"price": px * 1.02, "pct": 0.02, "direction": "up"},
                    "meta": {"side": "bullish_island"},
                }
            )
    return out


def _compression_patterns(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(close) < 10:
        return out
    ranges = high - low
    # NR7: narrowest range of last 7
    if len(ranges) >= 7 and ranges[-1] <= ranges[-7:].min() + 1e-12:
        out.append(
            {
                "pattern": "NR7",
                "rule_confidence": 0.75,
                "target_projection": {
                    "price": float(close[-1]),
                    "pct": 0.0,
                    "direction": "neutral",
                },
                "meta": {"range": float(ranges[-1])},
            }
        )
    # Inside bar
    if high[-1] <= high[-2] and low[-1] >= low[-2]:
        out.append(
            {
                "pattern": "Inside Bar",
                "rule_confidence": 0.78,
                "target_projection": {
                    "price": float(close[-1]),
                    "pct": 0.0,
                    "direction": "neutral",
                },
                "meta": {"mother_high": float(high[-2]), "mother_low": float(low[-2])},
            }
        )
    # Volatility squeeze: BB-like width vs ATR proxy
    rets = np.diff(close) / np.clip(close[:-1], 1e-9, None)
    rvol = float(np.std(rets[-10:])) if len(rets) >= 10 else float(np.std(rets) or 0)
    atr = float(np.mean(ranges[-14:]))
    if rvol < 0.012 or atr / max(close[-1], 1e-9) < 0.012:
        out.append(
            {
                "pattern": "Volatility Squeeze",
                "rule_confidence": 0.7,
                "target_projection": {
                    "price": float(close[-1] * (1.02 if close[-1] >= close[-5] else 0.98)),
                    "pct": 0.02 if close[-1] >= close[-5] else -0.02,
                    "direction": "up" if close[-1] >= close[-5] else "down",
                },
                "meta": {"realized_vol": rvol, "atr_pct": atr / max(close[-1], 1e-9)},
            }
        )
    return out


def _legacy_structures(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> list[dict[str, Any]]:
    """Keep prior API pattern names for compatibility."""
    out: list[dict[str, Any]] = []
    prior_high = float(high[:-1].max())
    if close[-1] > prior_high:
        vol_confirm = float(volume[-1]) > float(np.median(volume[:-1]) + 1e-9) * 1.2
        out.append(
            {
                "pattern": "range_breakout",
                "rule_confidence": 0.75 if vol_confirm else 0.55,
                "target_projection": {
                    "price": float(close[-1] + (close[-1] - prior_high)),
                    "pct": float((close[-1] - prior_high) / max(prior_high, 1e-9)),
                    "direction": "up",
                },
                "meta": {"level": prior_high, "volume_confirm": vol_confirm},
            }
        )
    hh = close[-1] > close[-5:].max() * 0.998 and close[-1] > close[0]
    hl = low[-1] > low[:-1].min()
    if hh and hl:
        out.append(
            {
                "pattern": "uptrend_structure",
                "rule_confidence": 0.7,
                "target_projection": {
                    "price": float(close[-1] * 1.02),
                    "pct": 0.02,
                    "direction": "up",
                },
                "meta": {},
            }
        )
    return out


# ---------------------------------------------------------------------------
# Candlestick library
# ---------------------------------------------------------------------------


def _detect_candlesticks(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    o, h, l, c = open_, high, low, close
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-9)
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    i = -1
    px = float(c[i])

    # Doji
    if body[i] / rng[i] < 0.1:
        out.append(_candle("Doji", 0.7, px, "neutral", spot=px))
    # Hammer
    if lower[i] > 2 * body[i] and upper[i] < body[i] * 0.5 and c[i] >= o[i]:
        out.append(_candle("Hammer", 0.72, px * 1.015, "up", spot=px))
    # Shooting star
    if upper[i] > 2 * body[i] and lower[i] < body[i] * 0.5 and c[i] <= o[i]:
        out.append(_candle("Shooting Star", 0.72, px * 0.985, "down", spot=px))
    # Engulfing
    if len(c) >= 2:
        if c[i] > o[i] and c[i - 1] < o[i - 1] and c[i] >= o[i - 1] and o[i] <= c[i - 1]:
            out.append(_candle("Bullish Engulfing", 0.76, px * 1.02, "up", spot=px))
        if c[i] < o[i] and c[i - 1] > o[i - 1] and c[i] <= o[i - 1] and o[i] >= c[i - 1]:
            out.append(_candle("Bearish Engulfing", 0.76, px * 0.98, "down", spot=px))
        # Harami
        if body[i] < body[i - 1] * 0.6 and max(c[i], o[i]) <= max(c[i - 1], o[i - 1]) and min(c[i], o[i]) >= min(
            c[i - 1], o[i - 1]
        ):
            out.append(_candle("Harami", 0.65, px, "neutral", spot=px))
        # Piercing / dark cloud
        mid_prev = (o[i - 1] + c[i - 1]) / 2
        if c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] < c[i - 1] and c[i] > mid_prev:
            out.append(_candle("Piercing Line", 0.7, px * 1.015, "up", spot=px))
        if c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] > c[i - 1] and c[i] < mid_prev:
            out.append(_candle("Dark Cloud Cover", 0.7, px * 0.985, "down", spot=px))
    if len(c) >= 3:
        # Morning / evening star
        if (
            c[i - 2] < o[i - 2]
            and body[i - 1] / rng[i - 1] < 0.3
            and c[i] > o[i]
            and c[i] > (o[i - 2] + c[i - 2]) / 2
        ):
            out.append(_candle("Morning Star", 0.74, px * 1.02, "up", spot=px))
        if (
            c[i - 2] > o[i - 2]
            and body[i - 1] / rng[i - 1] < 0.3
            and c[i] < o[i]
            and c[i] < (o[i - 2] + c[i - 2]) / 2
        ):
            out.append(_candle("Evening Star", 0.74, px * 0.98, "down", spot=px))
        # Three soldiers / crows
        if all(c[j] > o[j] for j in (-3, -2, -1)) and c[-1] > c[-2] > c[-3]:
            out.append(_candle("Three White Soldiers", 0.73, px * 1.025, "up", spot=px))
        if all(c[j] < o[j] for j in (-3, -2, -1)) and c[-1] < c[-2] < c[-3]:
            out.append(_candle("Three Black Crows", 0.73, px * 0.975, "down", spot=px))
    return out


def _candle(name: str, conf: float, target: float, direction: str, *, spot: float | None = None) -> dict[str, Any]:
    px = float(spot) if spot is not None and spot else float(target)
    return {
        "pattern": name,
        "rule_confidence": conf,
        "target_projection": {
            "price": float(target),
            "pct": float(target / px - 1.0) if px else 0.0,
            "direction": direction,
        },
        "meta": {"library": "candlestick"},
    }


def _swing_pivots(high: np.ndarray, low: np.ndarray, close: np.ndarray, order: int = 2) -> list[dict]:
    out: list[dict] = []
    n = len(close)
    for i in range(order, n - order):
        if high[i] == high[i - order : i + order + 1].max():
            out.append({"i": i, "type": "high", "price": float(high[i])})
        if low[i] == low[i - order : i + order + 1].min():
            out.append({"i": i, "type": "low", "price": float(low[i])})
    return out
