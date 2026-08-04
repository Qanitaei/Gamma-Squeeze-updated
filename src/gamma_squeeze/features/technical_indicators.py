"""Technical Indicators feature engine — trend, momentum, vol bands, flow, breadth, profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TechnicalIndicatorFeatures:
    symbol: str
    as_of: str
    spot: float

    # Moving averages / VWAP
    ema_12: float | None = None
    ema_26: float | None = None
    ema_50: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    vwap: float | None = None
    anchored_vwap: float | None = None

    # Volatility / channels
    atr: float | None = None
    bollinger_width: float | None = None
    bollinger_upper: float | None = None
    bollinger_lower: float | None = None
    keltner_upper: float | None = None
    keltner_lower: float | None = None
    donchian_upper: float | None = None
    donchian_lower: float | None = None

    # Momentum
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    adx: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None
    cci: float | None = None

    # Ichimoku / SuperTrend
    ichimoku_tenkan: float | None = None
    ichimoku_kijun: float | None = None
    ichimoku_span_a: float | None = None
    ichimoku_span_b: float | None = None
    ichimoku_cloud_top: float | None = None
    ichimoku_cloud_bottom: float | None = None
    supertrend: float | None = None
    supertrend_dir: float | None = None  # 1 long / -1 short

    # Volume / money flow
    obv: float | None = None
    money_flow_index: float | None = None
    chaikin_money_flow: float | None = None
    accumulation_distribution: float | None = None
    volume_delta: float | None = None
    cumulative_delta: float | None = None

    # Breadth / tape proxies (optional cross-asset inputs)
    tick_index: float | None = None
    advance_decline: float | None = None
    trin: float | None = None
    breadth: float | None = None

    # Profiles / liquidity
    market_profile_poc: float | None = None
    market_profile_vah: float | None = None
    market_profile_val: float | None = None
    volume_profile_poc: float | None = None
    volume_profile: list[dict[str, float]] = field(default_factory=list)
    liquidity_score: float | None = None

    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat_features(self) -> dict[str, Any]:
        d = self.to_dict()
        d.pop("volume_profile", None)
        d.pop("meta", None)
        return d

    def catalog_dict(self) -> dict[str, Any]:
        """Payload keyed by locked Technical Indicators product names."""
        d = self.to_dict()
        return {
            "symbol": d["symbol"],
            "as_of": d["as_of"],
            "spot": d["spot"],
            "ema": {"ema_12": d["ema_12"], "ema_26": d["ema_26"], "ema_50": d["ema_50"]},
            "sma": {"sma_20": d["sma_20"], "sma_50": d["sma_50"], "sma_200": d["sma_200"]},
            "vwap": d["vwap"],
            "anchored_vwap": d["anchored_vwap"],
            "atr": d["atr"],
            "rsi": d["rsi"],
            "macd": {
                "macd": d["macd"],
                "signal": d["macd_signal"],
                "hist": d["macd_hist"],
            },
            "adx": {
                "adx": d["adx"],
                "plus_di": d["plus_di"],
                "minus_di": d["minus_di"],
            },
            "cci": d["cci"],
            "keltner": {"upper": d["keltner_upper"], "lower": d["keltner_lower"]},
            "donchian": {"upper": d["donchian_upper"], "lower": d["donchian_lower"]},
            "bollinger_width": d["bollinger_width"],
            "bollinger": {
                "upper": d["bollinger_upper"],
                "lower": d["bollinger_lower"],
                "width": d["bollinger_width"],
            },
            "ichimoku": {
                "tenkan": d["ichimoku_tenkan"],
                "kijun": d["ichimoku_kijun"],
                "span_a": d["ichimoku_span_a"],
                "span_b": d["ichimoku_span_b"],
                "cloud_top": d["ichimoku_cloud_top"],
                "cloud_bottom": d["ichimoku_cloud_bottom"],
            },
            "supertrend": {"level": d["supertrend"], "direction": d["supertrend_dir"]},
            "obv": d["obv"],
            "money_flow_index": d["money_flow_index"],
            "chaikin_money_flow": d["chaikin_money_flow"],
            "accumulation_distribution": d["accumulation_distribution"],
            "volume_delta": d["volume_delta"],
            "cumulative_delta": d["cumulative_delta"],
            "tick_index": d["tick_index"],
            "advance_decline": d["advance_decline"],
            "trin": d["trin"],
            "breadth": d["breadth"],
            "market_profile": {
                "poc": d["market_profile_poc"],
                "vah": d["market_profile_vah"],
                "val": d["market_profile_val"],
            },
            "volume_profile": {
                "poc": d["volume_profile_poc"],
                "bins": d["volume_profile"],
            },
            "liquidity_score": d["liquidity_score"],
            "meta": d.get("meta") or {},
        }


# Locked Feature Engineering — Technical Indicators product catalog
TECHNICAL_INDICATORS_FIELDS: tuple[str, ...] = (
    "ema",
    "sma",
    "vwap",
    "anchored_vwap",
    "atr",
    "rsi",
    "macd",
    "adx",
    "cci",
    "keltner",
    "donchian",
    "bollinger_width",
    "ichimoku",
    "supertrend",
    "obv",
    "money_flow_index",
    "chaikin_money_flow",
    "accumulation_distribution",
    "volume_delta",
    "cumulative_delta",
    "tick_index",
    "advance_decline",
    "trin",
    "breadth",
    "market_profile",
    "volume_profile",
    "liquidity_score",
)

# Flat ML columns
TECHNICAL_INDICATOR_COLUMNS = [
    "ema_12",
    "ema_26",
    "ema_50",
    "sma_20",
    "sma_50",
    "sma_200",
    "vwap",
    "anchored_vwap",
    "atr",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "adx",
    "plus_di",
    "minus_di",
    "cci",
    "keltner_upper",
    "keltner_lower",
    "donchian_upper",
    "donchian_lower",
    "bollinger_width",
    "bollinger_upper",
    "bollinger_lower",
    "ichimoku_tenkan",
    "ichimoku_kijun",
    "ichimoku_span_a",
    "ichimoku_span_b",
    "ichimoku_cloud_top",
    "ichimoku_cloud_bottom",
    "supertrend",
    "supertrend_dir",
    "obv",
    "money_flow_index",
    "chaikin_money_flow",
    "accumulation_distribution",
    "volume_delta",
    "cumulative_delta",
    "tick_index",
    "advance_decline",
    "trin",
    "breadth",
    "market_profile_poc",
    "market_profile_vah",
    "market_profile_val",
    "volume_profile_poc",
    "liquidity_score",
]


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            if col == "volume":
                out[col] = 0.0
            elif col in ("high", "low") and "close" in out.columns:
                out[col] = out["close"]
            elif col == "open" and "close" in out.columns:
                out[col] = out["close"]
            else:
                out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["close"])


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(2, window // 2)).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return _true_range(df).rolling(window, min_periods=max(2, window // 2)).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    line = ema_fast - ema_slow
    sig = _ema(line, signal)
    hist = line - sig
    return line, sig, hist


def _adx(df: pd.DataFrame, window: int = 14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / window, adjust=False).mean()
    return adx, plus_di, minus_di


def _cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(window, min_periods=max(2, window // 2)).mean()
    mad = tp.rolling(window, min_periods=max(2, window // 2)).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = _sma(close, window)
    sd = close.rolling(window, min_periods=max(2, window // 2)).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, width


def _keltner(df: pd.DataFrame, window: int = 20, atr_mult: float = 1.5):
    mid = _ema(df["close"], window)
    atr = _atr(df, window)
    return mid + atr_mult * atr, mid, mid - atr_mult * atr


def _donchian(df: pd.DataFrame, window: int = 20):
    upper = df["high"].rolling(window, min_periods=max(2, window // 2)).max()
    lower = df["low"].rolling(window, min_periods=max(2, window // 2)).min()
    return upper, lower


def _ichimoku(df: pd.DataFrame):
    high, low = df["high"], df["low"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2.0).shift(26)
    return tenkan, kijun, span_a, span_b


def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    atr = _atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    prev_st = np.nan
    prev_dir = 1.0
    for i in range(len(df)):
        if i == 0 or not np.isfinite(atr.iloc[i]):
            st.iloc[i] = lower.iloc[i]
            direction.iloc[i] = 1.0
            prev_st, prev_dir = st.iloc[i], 1.0
            continue
        curr_upper = upper.iloc[i]
        curr_lower = lower.iloc[i]
        close = df["close"].iloc[i]
        if prev_dir >= 0:
            curr_lower = max(curr_lower, prev_st) if np.isfinite(prev_st) else curr_lower
            if close < curr_lower:
                prev_dir = -1.0
                prev_st = curr_upper
            else:
                prev_dir = 1.0
                prev_st = curr_lower
        else:
            curr_upper = min(curr_upper, prev_st) if np.isfinite(prev_st) else curr_upper
            if close > curr_upper:
                prev_dir = 1.0
                prev_st = curr_lower
            else:
                prev_dir = -1.0
                prev_st = curr_upper
        st.iloc[i] = prev_st
        direction.iloc[i] = prev_dir
    return st, direction


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def _mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(window, min_periods=max(2, window // 2)).sum()
    neg_sum = neg.rolling(window, min_periods=max(2, window // 2)).sum()
    mr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mr))


def _cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    mfv = mfm.fillna(0.0) * df["volume"]
    return mfv.rolling(window, min_periods=max(2, window // 2)).sum() / df["volume"].rolling(
        window, min_periods=max(2, window // 2)
    ).sum().replace(0, np.nan)


def _ad_line(df: pd.DataFrame) -> pd.Series:
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    return (clv.fillna(0.0) * df["volume"]).cumsum()


def _volume_delta(df: pd.DataFrame) -> pd.Series:
    """Proxy: up-bar volume positive, down-bar negative (tick tape unavailable)."""
    bar_sign = np.sign(df["close"] - df["open"])
    chg_sign = np.sign(df["close"].diff()).fillna(0.0)
    sign = bar_sign.where(bar_sign != 0, chg_sign).fillna(0.0)
    return sign * df["volume"]


def _vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vp = (tp * df["volume"]).cumsum()
    cum_v = df["volume"].cumsum().replace(0, np.nan)
    return cum_vp / cum_v


def _anchored_vwap(df: pd.DataFrame, anchor_idx: int = 0) -> pd.Series:
    sub = df.iloc[anchor_idx:].copy()
    tp = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    cum_vp = (tp * sub["volume"]).cumsum()
    cum_v = sub["volume"].cumsum().replace(0, np.nan)
    avwap = cum_vp / cum_v
    out = pd.Series(index=df.index, dtype=float)
    out.iloc[anchor_idx:] = avwap.values
    return out


def _volume_profile(df: pd.DataFrame, bins: int = 24) -> tuple[float | None, list[dict[str, float]], float | None, float | None]:
    if df.empty:
        return None, [], None, None
    low = float(df["low"].min())
    high = float(df["high"].max())
    if high <= low:
        high = low + 1e-6
    edges = np.linspace(low, high, bins + 1)
    profile = np.zeros(bins)
    for _, row in df.iterrows():
        c = float(row["close"])
        v = float(row["volume"] or 0)
        idx = int(np.clip(np.digitize([c], edges)[0] - 1, 0, bins - 1))
        profile[idx] += v
    records = [
        {"price_lo": float(edges[i]), "price_hi": float(edges[i + 1]), "volume": float(profile[i])}
        for i in range(bins)
    ]
    poc_i = int(np.argmax(profile)) if profile.sum() > 0 else 0
    poc = float((edges[poc_i] + edges[poc_i + 1]) / 2)
    # VAH/VAL ~ 70% volume area around POC
    order = list(np.argsort(profile)[::-1])
    target = 0.7 * profile.sum()
    covered = 0.0
    selected = set()
    for i in order:
        selected.add(int(i))
        covered += profile[i]
        if covered >= target:
            break
    if selected:
        vah = float(edges[max(selected) + 1])
        val = float(edges[min(selected)])
    else:
        vah = val = None
    return poc, records, vah, val


def _liquidity_score(df: pd.DataFrame, window: int = 20) -> float | None:
    if len(df) < 5:
        return None
    vol = df["volume"].tail(window)
    tr = _true_range(df).tail(window)
    # higher volume / tighter range => higher liquidity
    v = float(vol.mean()) if len(vol) else 0.0
    r = float(tr.mean()) if len(tr) else 0.0
    if v <= 0:
        return 0.0
    raw = v / (r * max(float(df["close"].iloc[-1]), 1e-9) + 1e-9)
    # squash to [0,1]
    return float(max(0.0, min(1.0, np.tanh(raw / 1e6))))


def _last(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)


def compute_technical_indicators(
    ohlcv: pd.DataFrame,
    *,
    symbol: str = "",
    as_of: str | None = None,
    anchor_date: str | None = None,
    breadth: dict[str, float] | None = None,
    lookback_profile: int = 60,
) -> TechnicalIndicatorFeatures:
    """Compute full technical indicator block from OHLCV bars."""
    df = _ensure_ohlcv(ohlcv)
    if as_of and as_of in df.index:
        df = df.loc[:as_of]
    if df.empty:
        return TechnicalIndicatorFeatures(symbol=symbol.upper(), as_of=as_of or "", spot=0.0)

    spot = float(df["close"].iloc[-1])
    as_of_date = as_of or str(df.index[-1])[:10]
    feats = TechnicalIndicatorFeatures(symbol=symbol.upper(), as_of=as_of_date, spot=spot)

    close = df["close"]
    feats.ema_12 = _last(_ema(close, 12))
    feats.ema_26 = _last(_ema(close, 26))
    feats.ema_50 = _last(_ema(close, 50))
    feats.sma_20 = _last(_sma(close, 20))
    feats.sma_50 = _last(_sma(close, 50))
    feats.sma_200 = _last(_sma(close, 200))
    feats.vwap = _last(_vwap(df))

    # Anchored VWAP: default to first bar of lookback window or explicit anchor
    if anchor_date and anchor_date in df.index:
        anchor_idx = list(df.index).index(anchor_date)
    else:
        anchor_idx = max(0, len(df) - lookback_profile)
    feats.anchored_vwap = _last(_anchored_vwap(df, anchor_idx=anchor_idx))

    feats.atr = _last(_atr(df, 14))
    bu, bm, bl, bw = _bollinger(close, 20)
    feats.bollinger_upper = _last(bu)
    feats.bollinger_lower = _last(bl)
    feats.bollinger_width = _last(bw)

    ku, km, kl = _keltner(df, 20)
    feats.keltner_upper = _last(ku)
    feats.keltner_lower = _last(kl)

    du, dl = _donchian(df, 20)
    feats.donchian_upper = _last(du)
    feats.donchian_lower = _last(dl)

    feats.rsi = _last(_rsi(close, 14))
    macd, signal, hist = _macd(close)
    feats.macd = _last(macd)
    feats.macd_signal = _last(signal)
    feats.macd_hist = _last(hist)

    adx, pdi, mdi = _adx(df, 14)
    feats.adx = _last(adx)
    feats.plus_di = _last(pdi)
    feats.minus_di = _last(mdi)
    feats.cci = _last(_cci(df, 20))

    tenkan, kijun, span_a, span_b = _ichimoku(df)
    feats.ichimoku_tenkan = _last(tenkan)
    feats.ichimoku_kijun = _last(kijun)
    feats.ichimoku_span_a = _last(span_a)
    feats.ichimoku_span_b = _last(span_b)
    if feats.ichimoku_span_a is not None and feats.ichimoku_span_b is not None:
        feats.ichimoku_cloud_top = max(feats.ichimoku_span_a, feats.ichimoku_span_b)
        feats.ichimoku_cloud_bottom = min(feats.ichimoku_span_a, feats.ichimoku_span_b)

    st, st_dir = _supertrend(df)
    feats.supertrend = _last(st)
    feats.supertrend_dir = _last(st_dir)

    feats.obv = _last(_obv(df))
    feats.money_flow_index = _last(_mfi(df, 14))
    feats.chaikin_money_flow = _last(_cmf(df, 20))
    feats.accumulation_distribution = _last(_ad_line(df))
    vdelta = _volume_delta(df)
    feats.volume_delta = _last(vdelta)
    feats.cumulative_delta = _last(vdelta.cumsum())

    # Breadth / tape — pass-through from optional market-structure feed
    b = breadth or {}
    feats.tick_index = b.get("tick_index")
    feats.advance_decline = b.get("advance_decline")
    feats.trin = b.get("trin")
    feats.breadth = b.get("breadth")
    # Single-name proxies if breadth unavailable
    if feats.breadth is None and len(df) >= 20:
        up = float((close.diff().tail(20) > 0).mean())
        feats.breadth = up
        feats.advance_decline = float((close.diff().tail(20) > 0).sum() - (close.diff().tail(20) < 0).sum())
        feats.tick_index = float(np.clip((close.pct_change().tail(20).mean() or 0.0) * 10000, -100, 100))
        # TRIN proxy: down_vol/up_vol over advancing/declining days
        chg = close.diff()
        up_vol = float(df.loc[chg > 0, "volume"].tail(20).sum())
        down_vol = float(df.loc[chg < 0, "volume"].tail(20).sum())
        adv = max(float((chg.tail(20) > 0).sum()), 1.0)
        dec = max(float((chg.tail(20) < 0).sum()), 1.0)
        feats.trin = float((down_vol / max(up_vol, 1.0)) / (dec / adv)) if up_vol > 0 else None

    profile_df = df.tail(lookback_profile)
    poc, records, vah, val = _volume_profile(profile_df, bins=24)
    feats.volume_profile_poc = poc
    feats.volume_profile = records
    # Market profile approximated via same volume-at-price distribution on daily bars
    feats.market_profile_poc = poc
    feats.market_profile_vah = vah
    feats.market_profile_val = val
    feats.liquidity_score = _liquidity_score(df, window=20)

    feats.meta = {
        "n_bars": int(len(df)),
        "anchor_idx": int(anchor_idx),
        "lookback_profile": lookback_profile,
    }
    return feats
