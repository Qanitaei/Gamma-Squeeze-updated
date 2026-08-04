"""Macro Features engine — Fed, inflation, growth, curve, FX, vol, credit from Fred-Economic-data KV."""

from __future__ import annotations

import json
import ssl
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from gamma_squeeze.ingest.macro_client import (
    fetch_calendar_near,
    fetch_fred_history,
    fetch_fred_value,
    list_fred_series,
)

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


# Canonical feature → ordered FRED series candidates in Fred-Economic-data
MACRO_SERIES_MAP: dict[str, list[str]] = {
    "fed_funds": ["FEDFUNDS"],
    "cpi": ["CPIAUCSL"],
    "core_cpi": ["CPILFESL"],
    "ppi": ["PPIACO", "WPSFD4131", "PPIFIS"],
    "payrolls": ["PAYEMS"],
    "pmi": ["GACDISA066MSFRBNY", "NAPM"],  # Empire State PMI proxy; NAPM if synced
    "gdp": ["A191RL1Q225SBEA", "GDP"],  # prefer real GDP growth QoQ SAAR
    "consumer_sentiment": ["UMCSENT"],
    "treasury_2y": ["DGS2", "GS2"],
    "treasury_10y": ["DGS10", "GS10"],
    "treasury_30y": ["DGS30"],
    "yield_spread": ["T10Y2Y"],
    "yield_spread_10y3m": ["T10Y3M"],
    "dollar_index": ["DTWEXBGS"],
    "usdjpy": ["DEXJPUS"],
    "vix": ["VIXCLS"],
    "credit_spread": ["BAA10Y"],
    "corporate_bond_spread": ["BAMLC0A0CM"],
    "high_yield_spread": ["BAMLH0A0HYM2"],
    "weekly_economic_index": ["WEI"],
    "financial_stress": ["STLFSI4"],
}


@dataclass
class MacroFeatures:
    as_of: str

    fed_funds: float | None = None
    cpi: float | None = None
    core_cpi: float | None = None
    ppi: float | None = None
    payrolls: float | None = None
    pmi: float | None = None
    gdp: float | None = None
    consumer_sentiment: float | None = None

    treasury_2y: float | None = None
    treasury_10y: float | None = None
    treasury_30y: float | None = None
    treasury_curve: dict[str, float | None] = field(default_factory=dict)
    yield_spread: float | None = None
    yield_spread_10y3m: float | None = None

    dollar_index: float | None = None
    usdjpy: float | None = None

    move: float | None = None
    vix: float | None = None
    vvix: float | None = None

    economic_surprise_index: float | None = None
    credit_spread: float | None = None
    corporate_bond_spread: float | None = None
    high_yield_spread: float | None = None

    # YoY / momentum helpers
    cpi_yoy: float | None = None
    ppi_yoy: float | None = None
    payrolls_mom: float | None = None

    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat_features(self) -> dict[str, Any]:
        d = self.to_dict()
        curve = d.pop("treasury_curve", {}) or {}
        d.pop("meta", None)
        # flatten curve tenors already present as treasury_* fields
        for k, v in curve.items():
            key = f"treasury_curve_{k}"
            if key not in d:
                d[key] = v
        return d

    def catalog_dict(self) -> dict[str, Any]:
        """Payload keyed by locked Macro Features product names."""
        d = self.to_dict()
        return {
            "as_of": d["as_of"],
            "fed_funds": d["fed_funds"],
            "cpi": d["cpi"],
            "ppi": d["ppi"],
            "payrolls": d["payrolls"],
            "pmi": d["pmi"],
            "gdp": d["gdp"],
            "consumer_sentiment": d["consumer_sentiment"],
            "treasury_curve": d["treasury_curve"],
            "yield_spread": d["yield_spread"],
            "dollar_index": d["dollar_index"],
            "usdjpy": d["usdjpy"],
            "move": d["move"],
            "vix": d["vix"],
            "vvix": d["vvix"],
            "economic_surprise_index": d["economic_surprise_index"],
            "credit_spread": d["credit_spread"],
            "corporate_bond_spread": d["corporate_bond_spread"],
            "high_yield_spread": d["high_yield_spread"],
            # helpers retained for ML
            "core_cpi": d["core_cpi"],
            "yield_spread_10y3m": d["yield_spread_10y3m"],
            "cpi_yoy": d["cpi_yoy"],
            "ppi_yoy": d["ppi_yoy"],
            "payrolls_mom": d["payrolls_mom"],
            "meta": d.get("meta") or {},
        }


# Locked Feature Engineering — Macro Features product catalog
MACRO_FEATURES_FIELDS: tuple[str, ...] = (
    "fed_funds",
    "cpi",
    "ppi",
    "payrolls",
    "pmi",
    "gdp",
    "consumer_sentiment",
    "treasury_curve",
    "yield_spread",
    "dollar_index",
    "usdjpy",
    "move",
    "vix",
    "vvix",
    "economic_surprise_index",
    "credit_spread",
    "corporate_bond_spread",
    "high_yield_spread",
)

# Flat ML columns (curve flattened separately in flat_features)
MACRO_FEATURE_COLUMNS = [
    "fed_funds",
    "cpi",
    "core_cpi",
    "ppi",
    "payrolls",
    "pmi",
    "gdp",
    "consumer_sentiment",
    "treasury_2y",
    "treasury_10y",
    "treasury_30y",
    "yield_spread",
    "yield_spread_10y3m",
    "dollar_index",
    "usdjpy",
    "move",
    "vix",
    "vvix",
    "economic_surprise_index",
    "credit_spread",
    "corporate_bond_spread",
    "high_yield_spread",
    "cpi_yoy",
    "ppi_yoy",
    "payrolls_mom",
]


def _resolve_series(feature: str, available: set[str] | None = None) -> str | None:
    for sid in MACRO_SERIES_MAP.get(feature, []):
        if available is None or sid in available or not available:
            return sid
    cands = MACRO_SERIES_MAP.get(feature, [])
    return cands[0] if cands else None


def _yoy(hist: list[tuple[str, float]], months: int = 12) -> float | None:
    if len(hist) < months + 1:
        return None
    latest = hist[-1][1]
    # approximate: look back ~months observations for monthly series
    base = hist[-(months + 1)][1]
    if base == 0:
        return None
    return (latest / base) - 1.0


def _mom(hist: list[tuple[str, float]]) -> float | None:
    if len(hist) < 2:
        return None
    prev = hist[-2][1]
    if prev == 0:
        return None
    return (hist[-1][1] / prev) - 1.0


def _zscore_last(hist: list[tuple[str, float]], window: int = 52) -> float | None:
    if len(hist) < max(8, window // 4):
        return None
    vals = np.array([v for _, v in hist[-window:]], dtype=float)
    diffs = np.diff(vals)
    if len(diffs) < 4:
        return None
    mu = float(np.mean(diffs[:-1])) if len(diffs) > 1 else 0.0
    sd = float(np.std(diffs[:-1], ddof=1)) if len(diffs) > 2 else 0.0
    if sd <= 1e-12:
        return 0.0
    return float((diffs[-1] - mu) / sd)


def _fetch_cboe_index(symbol: str) -> float | None:
    """Delayed CBOE quote for VIX / VVIX when FRED key absent."""
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/quotes/_{symbol.upper()}.json"
    req = Request(url, headers={"User-Agent": "GammaSqueezePlatform/0.1"})
    try:
        with urlopen(req, timeout=20, context=_SSL_CTX) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") or {}
        for key in ("current_price", "last", "close"):
            if data.get(key) is not None:
                return float(data[key])
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def _calendar_surprise(as_of: str) -> float | None:
    """Mean standardized (actual − forecast) over nearby calendar prints when available."""
    try:
        events = fetch_calendar_near(as_of, look_ahead_days=0)
    except Exception:  # noqa: BLE001
        return None
    scores: list[float] = []
    for e in events:
        actual, forecast = e.get("actual"), e.get("forecast")
        try:
            a = float(str(actual).replace("%", "").replace(",", ""))
            f = float(str(forecast).replace("%", "").replace(",", ""))
        except (TypeError, ValueError):
            continue
        denom = abs(f) if abs(f) > 1e-9 else 1.0
        scores.append((a - f) / denom)
    if not scores:
        return None
    return float(np.mean(scores))


def compute_macro_features(
    *,
    as_of: str | None = None,
    include_external_vol: bool = True,
    base_url: str | None = None,
) -> MacroFeatures:
    """
    Build macro feature block from Fred-Economic-data KV (with local mirror fallback).

    MOVE / VVIX are not standard FRED series; VVIX uses CBOE delayed when enabled.
    Economic Surprise prefers calendar actual−forecast, else WEI change z-score.
    """
    as_of_date = (as_of or "")[:10]
    available: set[str] = set()
    try:
        available = set(list_fred_series(base_url=base_url))
    except Exception:  # noqa: BLE001
        available = set()

    origins: dict[str, str] = {}
    series_used: dict[str, str | None] = {}

    def _value(feature: str) -> float | None:
        sid = _resolve_series(feature, available or None)
        series_used[feature] = sid
        if not sid:
            return None
        val = fetch_fred_value(sid, as_of=as_of_date or None, base_url=base_url)
        if val is not None:
            origins[feature] = sid
        return val

    feats = MacroFeatures(as_of=as_of_date)
    feats.fed_funds = _value("fed_funds")
    feats.cpi = _value("cpi")
    feats.core_cpi = _value("core_cpi")
    feats.ppi = _value("ppi")
    feats.payrolls = _value("payrolls")
    feats.pmi = _value("pmi")
    feats.gdp = _value("gdp")
    feats.consumer_sentiment = _value("consumer_sentiment")
    feats.treasury_2y = _value("treasury_2y")
    feats.treasury_10y = _value("treasury_10y")
    feats.treasury_30y = _value("treasury_30y")
    feats.yield_spread = _value("yield_spread")
    feats.yield_spread_10y3m = _value("yield_spread_10y3m")
    # Derive 10y-2y if T10Y2Y missing
    if feats.yield_spread is None and feats.treasury_10y is not None and feats.treasury_2y is not None:
        feats.yield_spread = feats.treasury_10y - feats.treasury_2y
        origins["yield_spread"] = "DGS10-DGS2"

    feats.treasury_curve = {
        "2y": feats.treasury_2y,
        "10y": feats.treasury_10y,
        "30y": feats.treasury_30y,
        "spread_10y2y": feats.yield_spread,
        "spread_10y3m": feats.yield_spread_10y3m,
    }

    feats.dollar_index = _value("dollar_index")
    feats.usdjpy = _value("usdjpy")
    feats.vix = _value("vix")
    feats.credit_spread = _value("credit_spread")
    feats.corporate_bond_spread = _value("corporate_bond_spread")
    feats.high_yield_spread = _value("high_yield_spread")

    # YoY / MoM
    cpi_hist = fetch_fred_history(origins.get("cpi", "CPIAUCSL"), as_of=as_of_date or None, base_url=base_url)
    ppi_sid = origins.get("ppi") or _resolve_series("ppi", available or None) or "PPIACO"
    ppi_hist = fetch_fred_history(ppi_sid, as_of=as_of_date or None, base_url=base_url)
    pay_hist = fetch_fred_history(origins.get("payrolls", "PAYEMS"), as_of=as_of_date or None, base_url=base_url)
    feats.cpi_yoy = _yoy(cpi_hist, 12)
    feats.ppi_yoy = _yoy(ppi_hist, 12)
    feats.payrolls_mom = _mom(pay_hist)

    # Economic surprise: calendar first, else WEI z-score, else -STLFSI4
    if as_of_date:
        feats.economic_surprise_index = _calendar_surprise(as_of_date)
        if feats.economic_surprise_index is not None:
            origins["economic_surprise_index"] = "calendar_actual_vs_forecast"
    if feats.economic_surprise_index is None:
        wei_hist = fetch_fred_history("WEI", as_of=as_of_date or None, base_url=base_url)
        feats.economic_surprise_index = _zscore_last(wei_hist, window=52)
        if feats.economic_surprise_index is not None:
            origins["economic_surprise_index"] = "WEI_delta_zscore"
    if feats.economic_surprise_index is None:
        stress = fetch_fred_value("STLFSI4", as_of=as_of_date or None, base_url=base_url)
        if stress is not None:
            feats.economic_surprise_index = float(-stress)
            origins["economic_surprise_index"] = "-STLFSI4"

    # External vol / FX fallbacks (CBOE delayed, structure adapters, OHLCV)
    if include_external_vol:
        if feats.vix is None:
            feats.vix = _fetch_cboe_index("VIX")
            if feats.vix is not None:
                origins["vix"] = "cboe_delayed"
        feats.vvix = _fetch_cboe_index("VVIX")
        if feats.vvix is not None:
            origins["vvix"] = "cboe_delayed"

        feats.move = fetch_fred_value("MOVE", as_of=as_of_date or None, base_url=base_url)
        if feats.move is not None:
            origins["move"] = "MOVE"
        else:
            # ICE BofA MOVE via OHLCV / structure candidates
            try:
                from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv

                for sym in ("^MOVE", "MOVE", "MOV"):
                    try:
                        df = fetch_daily_ohlcv(sym)
                        if df is None or df.empty:
                            continue
                        if as_of_date and as_of_date in df.index:
                            val = float(df.loc[:as_of_date, "close"].iloc[-1])
                        else:
                            val = float(df["close"].iloc[-1])
                        # ICE BofA MOVE typically trades ~50–200; skip bogus tickers
                        if not (20.0 <= val <= 400.0):
                            continue
                        feats.move = val
                        origins["move"] = f"ohlcv:{sym}"
                        break
                    except Exception:  # noqa: BLE001
                        continue
            except Exception:  # noqa: BLE001
                pass

        if feats.dollar_index is None or feats.usdjpy is None:
            try:
                from gamma_squeeze.data_sources.structure.fx import FXStructureAdapter

                fx = (FXStructureAdapter().fetch().get("data") or {}).get("fx") or {}
                # Accept only true index/pair origins — not ETF proxies (UUP/FXY)
                dxy = fx.get("DXY") or {}
                if (
                    feats.dollar_index is None
                    and dxy.get("value") is not None
                    and str(dxy.get("origin") or "").upper() in {"DXY", "DX-Y.NYB", "^DXY"}
                ):
                    feats.dollar_index = float(dxy["value"])
                    origins["dollar_index"] = f"structure.fx:{dxy.get('origin')}"
                usdjpy = fx.get("USDJPY") or {}
                origin_fx = str(usdjpy.get("origin") or "").upper()
                if (
                    feats.usdjpy is None
                    and usdjpy.get("value") is not None
                    and origin_fx in {"USDJPY", "USD/JPY", "JPY=X"}
                ):
                    feats.usdjpy = float(usdjpy["value"])
                    origins["usdjpy"] = f"structure.fx:{usdjpy.get('origin')}"
            except Exception:  # noqa: BLE001
                pass
    else:
        feats.move = fetch_fred_value("MOVE", as_of=as_of_date or None, base_url=base_url)

    if not as_of_date:
        # stamp with newest available observation date across core series
        for sid in ("DGS10", "VIXCLS", "FEDFUNDS", "CPIAUCSL"):
            hist = fetch_fred_history(sid, base_url=base_url)
            if hist:
                feats.as_of = hist[-1][0]
                break

    feats.meta = {
        "namespace": "Fred-Economic-data",
        "origins": origins,
        "series_used": series_used,
        "available_series_count": len(available),
        "pmi_note": "GACDISA066MSFRBNY Empire State Manufacturing Index used as PMI proxy when NAPM absent",
        "move_note": "MOVE is not a standard FRED series; populate KV key MOVE to enable",
    }
    return feats
