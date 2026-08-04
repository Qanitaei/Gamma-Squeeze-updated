"""Options Metrics feature engine — OI/volume, IV surface, vol ranks, RN probabilities."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

R = 0.04


@dataclass
class OptionsMetricsFeatures:
    symbol: str
    as_of: str
    spot: float

    # OI / volume
    open_interest: float = 0.0
    call_open_interest: float = 0.0
    put_open_interest: float = 0.0
    oi_velocity: float = 0.0  # ΔOI / prior OI
    oi_delta: float = 0.0  # absolute ΔOI
    volume_oi_ratio: float = 0.0
    call_volume: float = 0.0
    put_volume: float = 0.0
    total_volume: float = 0.0
    put_call_ratio: float = 0.0  # OI-based
    put_call_volume_ratio: float = 0.0

    # Surface shape
    skew: float | None = None  # 25Δ put IV − 25Δ call IV (approx)
    smile: float | None = None  # wing avg IV − ATM IV
    term_structure: float | None = None  # far ATM IV − near ATM IV
    atm_iv: float | None = None
    iv_rank: float | None = None
    iv_percentile: float | None = None

    # Vol measures
    historical_volatility: float | None = None
    realized_volatility: float | None = None
    forward_volatility: float | None = None
    expected_move: float | None = None  # 1σ dollar move to nearest expiry
    expected_move_pct: float | None = None

    # Probabilities (ATM / representative)
    probability_itm: float | None = None
    probability_touch: float | None = None

    # Risk-neutral density
    risk_neutral_density: list[dict[str, float]] = field(default_factory=list)

    # Aux surface samples
    smile_curve: list[dict[str, float]] = field(default_factory=list)
    term_curve: list[dict[str, float]] = field(default_factory=list)

    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat_features(self) -> dict[str, Any]:
        d = self.to_dict()
        d.pop("risk_neutral_density", None)
        d.pop("smile_curve", None)
        d.pop("term_curve", None)
        d.pop("meta", None)
        return d


# Locked Feature Engineering — Options Metrics field catalog (product names)
OPTIONS_METRICS_FIELDS: tuple[str, ...] = (
    "open_interest",
    "oi_velocity",
    "oi_delta",
    "volume_oi_ratio",
    "call_volume",
    "put_volume",
    "put_call_ratio",
    "skew",
    "smile",
    "term_structure",
    "iv_rank",
    "iv_percentile",
    "historical_volatility",
    "realized_volatility",
    "forward_volatility",
    "expected_move",
    "probability_itm",
    "probability_touch",
    "risk_neutral_density",
)

# Flat ML columns (scalars; curves excluded)
OPTIONS_METRIC_COLUMNS = [
    "open_interest",
    "call_open_interest",
    "put_open_interest",
    "oi_velocity",
    "oi_delta",
    "volume_oi_ratio",
    "call_volume",
    "put_volume",
    "total_volume",
    "put_call_ratio",
    "put_call_volume_ratio",
    "skew",
    "smile",
    "term_structure",
    "atm_iv",
    "iv_rank",
    "iv_percentile",
    "historical_volatility",
    "realized_volatility",
    "forward_volatility",
    "expected_move",
    "expected_move_pct",
    "probability_itm",
    "probability_touch",
]


def _contracts(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    if matrix.get("contracts"):
        return list(matrix["contracts"])
    rows: list[dict[str, Any]] = []
    for exp_block in matrix.get("by_expiration", []):
        for strike_block in exp_block.get("strikes", []):
            for side in ("call", "put"):
                c = strike_block.get(side)
                if c:
                    rows.append(c)
    return rows


def _is_call(c: dict[str, Any]) -> bool:
    return str(c.get("put_call") or c.get("side") or "call").lower().startswith("c")


def _norm_iv(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    try:
        iv = float(raw)
    except (TypeError, ValueError):
        return None
    if iv > 5:
        iv /= 100.0
    if iv <= 0 or iv > 3:
        return None
    return iv


def _t_years(c: dict[str, Any], now: datetime) -> float:
    dte = c.get("days_to_expiration")
    if dte is not None:
        try:
            return max(float(dte) / 365.0, 1.0 / 365.0)
        except (TypeError, ValueError):
            pass
    exp = c.get("expiration_date")
    if not exp:
        return 30 / 365
    try:
        dt = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max((dt - now).total_seconds() / (365.25 * 86400), 1 / 365)
    except ValueError:
        return 30 / 365


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def realized_vol_from_ohlcv(ohlcv: pd.DataFrame, *, window: int = 20) -> float | None:
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        return None
    rets = ohlcv["close"].astype(float).pct_change().dropna().tail(window)
    if len(rets) < max(5, window // 2):
        return None
    return float(rets.std() * math.sqrt(252))


def historical_vol_from_ohlcv(ohlcv: pd.DataFrame, *, window: int = 252) -> float | None:
    """Close-to-close HV over a longer window (defaults to ~1y)."""
    return realized_vol_from_ohlcv(ohlcv, window=window)


def iv_rank_percentile(history: list[float], current: float) -> tuple[float | None, float | None]:
    """IV Rank = (cur - min) / (max - min); IV Percentile = % of days IV was below current."""
    series = [float(x) for x in history if x is not None and np.isfinite(x)]
    if not series:
        return None, None
    lo, hi = min(series), max(series)
    rank = (current - lo) / (hi - lo) if hi > lo else 0.5
    pct = sum(1 for x in series if x < current) / len(series)
    return float(rank), float(pct)


def probability_itm_call(spot: float, strike: float, t: float, iv: float, r: float = R) -> float:
    if spot <= 0 or strike <= 0 or t <= 0 or iv <= 0:
        return 0.0
    d2 = (math.log(spot / strike) + (r - 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    return float(_norm_cdf(d2))


def probability_touch(spot: float, barrier: float, t: float, iv: float) -> float:
    """Approx. prob of touching barrier before T (driftless BM barrier formula)."""
    if spot <= 0 or barrier <= 0 or t <= 0 or iv <= 0:
        return 0.0
    if abs(barrier - spot) < 1e-12:
        return 1.0
    # P(tau < T) ≈ 2 * Φ(-|ln(H/S)| / (σ√T)) for driftless case
    x = abs(math.log(barrier / spot)) / (iv * math.sqrt(t))
    return float(min(1.0, max(0.0, 2.0 * _norm_cdf(-x))))


def risk_neutral_density_from_calls(
    spot: float,
    strikes: list[float],
    call_mids: list[float],
    t: float,
    r: float = R,
) -> list[dict[str, float]]:
    """Breeden-Litzenberger: q(K) ≈ e^{rT} * d²C/dK² via finite differences."""
    if len(strikes) < 3 or t <= 0:
        return []
    pairs = sorted(zip(strikes, call_mids), key=lambda x: x[0])
    ks = [p[0] for p in pairs]
    cs = [p[1] for p in pairs]
    dens: list[dict[str, float]] = []
    disc = math.exp(r * t)
    for i in range(1, len(ks) - 1):
        dK1 = ks[i] - ks[i - 1]
        dK2 = ks[i + 1] - ks[i]
        if dK1 <= 0 or dK2 <= 0:
            continue
        # second derivative
        d2 = ((cs[i + 1] - cs[i]) / dK2 - (cs[i] - cs[i - 1]) / dK1) / (0.5 * (dK1 + dK2))
        q = max(disc * d2, 0.0)
        dens.append({"strike": float(ks[i]), "density": float(q)})
    # normalize area roughly
    if dens:
        # trapezoid weights
        area = 0.0
        for i in range(len(dens) - 1):
            area += 0.5 * (dens[i]["density"] + dens[i + 1]["density"]) * (
                dens[i + 1]["strike"] - dens[i]["strike"]
            )
        if area > 0:
            for row in dens:
                row["density"] = float(row["density"] / area)
    return dens


def compute_options_metrics(
    matrix: dict[str, Any],
    *,
    as_of: str = "",
    ohlcv: pd.DataFrame | None = None,
    prior_oi: float | None = None,
    atm_iv_history: list[float] | None = None,
) -> OptionsMetricsFeatures:
    symbol = str(matrix.get("symbol", "")).upper()
    spot = float(matrix.get("underlying_price") or 0.0)
    as_of_date = as_of or str(matrix.get("as_of_date") or "")[:10]
    feats = OptionsMetricsFeatures(symbol=symbol, as_of=as_of_date, spot=spot)
    contracts = _contracts(matrix)
    now = datetime.now(timezone.utc)
    if spot <= 0 or not contracts:
        feats.meta["error"] = "missing_spot_or_contracts"
        return feats

    call_oi = put_oi = 0.0
    call_vol = put_vol = 0.0

    # Surface buckets: by expiry ATM IV, and moneyness IVs for nearest expiry
    by_exp: dict[str, list[tuple[float, float, bool, float]]] = {}
    # (strike, iv, is_call, t)

    for c in contracts:
        oi = float(c.get("open_interest") or 0)
        vol = float(c.get("total_volume") or 0)
        is_call = _is_call(c)
        if is_call:
            call_oi += oi
            call_vol += vol
        else:
            put_oi += oi
            put_vol += vol

        iv = _norm_iv(c.get("volatility") or c.get("implied_volatility"))
        strike = float(c.get("strike_price") or 0)
        if iv is None or strike <= 0:
            continue
        t = _t_years(c, now)
        exp_key = str(c.get("expiration_date") or round(t, 4))[:10]
        by_exp.setdefault(exp_key, []).append((strike, iv, is_call, t))

    total_oi = call_oi + put_oi
    total_vol = call_vol + put_vol
    feats.open_interest = float(total_oi)
    feats.call_open_interest = float(call_oi)
    feats.put_open_interest = float(put_oi)
    feats.call_volume = float(call_vol)
    feats.put_volume = float(put_vol)
    feats.total_volume = float(total_vol)
    feats.volume_oi_ratio = float(total_vol / total_oi) if total_oi > 0 else 0.0
    feats.put_call_ratio = float(put_oi / call_oi) if call_oi > 0 else 0.0
    feats.put_call_volume_ratio = float(put_vol / call_vol) if call_vol > 0 else 0.0

    if prior_oi is not None and prior_oi > 0:
        feats.oi_delta = float(total_oi - prior_oi)
        feats.oi_velocity = float((total_oi - prior_oi) / prior_oi)
    else:
        feats.oi_delta = 0.0
        feats.oi_velocity = 0.0

    # --- Term structure & ATM IV per expiry ---
    term_points: list[tuple[float, float]] = []  # (t_years, atm_iv)
    nearest_exp_key = None
    nearest_t = 1e9
    nearest_rows: list[tuple[float, float, bool, float]] = []

    for exp_key, rows in by_exp.items():
        if not rows:
            continue
        t_med = float(np.median([r[3] for r in rows]))
        # ATM = closest strike to spot (average call/put IV if both)
        best = min(rows, key=lambda r: abs(r[0] - spot))
        atm_iv = best[1]
        # refine: average IVs within 1% of spot
        near = [r[1] for r in rows if abs(r[0] - spot) / spot <= 0.01]
        if near:
            atm_iv = float(np.mean(near))
        term_points.append((t_med, atm_iv))
        feats.term_curve.append({"expiration": exp_key, "t_years": t_med, "atm_iv": atm_iv})
        if t_med < nearest_t:
            nearest_t = t_med
            nearest_exp_key = exp_key
            nearest_rows = rows

    term_points.sort(key=lambda x: x[0])
    if term_points:
        feats.atm_iv = float(term_points[0][1])
    if len(term_points) >= 2:
        feats.term_structure = float(term_points[-1][1] - term_points[0][1])

    # --- Skew / Smile on nearest expiry ---
    if nearest_rows:
        # build strike -> avg IV
        strike_iv: dict[float, list[float]] = {}
        call_ivs: list[tuple[float, float]] = []
        put_ivs: list[tuple[float, float]] = []
        for strike, iv, is_call, _t in nearest_rows:
            strike_iv.setdefault(strike, []).append(iv)
            if is_call:
                call_ivs.append((strike, iv))
            else:
                put_ivs.append((strike, iv))

        curve = sorted(
            ((k, float(np.mean(v))) for k, v in strike_iv.items()),
            key=lambda x: x[0],
        )
        feats.smile_curve = [{"strike": k, "iv": iv, "moneyness": (k / spot - 1.0)} for k, iv in curve]

        atm = feats.atm_iv or (curve[len(curve) // 2][1] if curve else None)
        if atm is not None and curve:
            # wings: |m| > 5%
            wings = [iv for k, iv in curve if abs(k / spot - 1.0) >= 0.05]
            if wings:
                feats.smile = float(np.mean(wings) - atm)

        # 25-delta approx via 25% OTM put/call
        put_25 = [iv for k, iv in put_ivs if 0.20 <= (1.0 - k / spot) <= 0.30] or [
            iv for k, iv in put_ivs if k < spot
        ]
        call_25 = [iv for k, iv in call_ivs if 0.20 <= (k / spot - 1.0) <= 0.30] or [
            iv for k, iv in call_ivs if k > spot
        ]
        if put_25 and call_25:
            feats.skew = float(np.mean(put_25) - np.mean(call_25))
        elif curve and atm is not None:
            # fallback: 90% strike IV - 110% strike IV
            lo = min(curve, key=lambda kv: abs(kv[0] / spot - 0.9))
            hi = min(curve, key=lambda kv: abs(kv[0] / spot - 1.1))
            feats.skew = float(lo[1] - hi[1])

    # --- HV / RV ---
    if ohlcv is not None and not ohlcv.empty:
        feats.realized_volatility = realized_vol_from_ohlcv(ohlcv, window=20)
        feats.historical_volatility = historical_vol_from_ohlcv(ohlcv, window=252)
        # truncate history to as_of if possible
        if as_of_date and as_of_date in ohlcv.index:
            hist = ohlcv.loc[:as_of_date]
            feats.realized_volatility = realized_vol_from_ohlcv(hist, window=20)
            feats.historical_volatility = historical_vol_from_ohlcv(hist, window=252)

    # --- IV rank / percentile ---
    if feats.atm_iv is not None:
        hist = list(atm_iv_history or [])
        if feats.atm_iv not in hist:
            hist = hist + [feats.atm_iv]
        feats.iv_rank, feats.iv_percentile = iv_rank_percentile(hist, feats.atm_iv)

    # --- Forward vol (calendar variance) ---
    if len(term_points) >= 2:
        t1, iv1 = term_points[0]
        t2, iv2 = term_points[1]
        if t2 > t1 > 0:
            var1 = iv1 * iv1 * t1
            var2 = iv2 * iv2 * t2
            fwd_var = (var2 - var1) / (t2 - t1)
            feats.forward_volatility = float(math.sqrt(max(fwd_var, 0.0)))

    # --- Expected move (nearest expiry, 1σ) ---
    if feats.atm_iv is not None and nearest_t < 1e8:
        em_pct = feats.atm_iv * math.sqrt(nearest_t)
        feats.expected_move_pct = float(em_pct)
        feats.expected_move = float(spot * em_pct)

    # --- Probability ITM / Touch (ATM strike) ---
    if feats.atm_iv is not None and nearest_t < 1e8:
        atm_strike = min(
            (r[0] for r in nearest_rows),
            key=lambda k: abs(k - spot),
            default=spot,
        )
        feats.probability_itm = probability_itm_call(spot, float(atm_strike), nearest_t, feats.atm_iv)
        # touch ± expected move barrier
        barrier = spot + (feats.expected_move or 0.0)
        feats.probability_touch = probability_touch(spot, barrier, nearest_t, feats.atm_iv)

    # --- Risk-neutral density from nearest expiry calls ---
    if nearest_rows and nearest_t < 1e8:
        call_strikes: list[float] = []
        call_mids: list[float] = []
        # Prefer quoted mids from the original matrix when present
        mid_by_strike: dict[float, list[float]] = {}
        for c in contracts:
            if not _is_call(c):
                continue
            strike = float(c.get("strike_price") or 0)
            if strike <= 0:
                continue
            exp_key = str(c.get("expiration_date") or "")[:10]
            if nearest_exp_key and exp_key and exp_key != nearest_exp_key:
                continue
            mid = c.get("mid") or c.get("mark") or c.get("last") or c.get("close")
            if mid not in (None, ""):
                try:
                    mid_by_strike.setdefault(strike, []).append(float(mid))
                except (TypeError, ValueError):
                    pass
        for strike, iv, is_call, _t in nearest_rows:
            if not is_call:
                continue
            if strike in mid_by_strike:
                call_strikes.append(strike)
                call_mids.append(float(np.mean(mid_by_strike[strike])))
                continue
            # BS call mid proxy from IV
            t = nearest_t
            d1 = (math.log(spot / strike) + (R + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
            d2 = d1 - iv * math.sqrt(t)
            call = spot * _norm_cdf(d1) - strike * math.exp(-R * t) * _norm_cdf(d2)
            call_strikes.append(strike)
            call_mids.append(max(call, 0.0))
        if call_strikes:
            df = pd.DataFrame({"k": call_strikes, "c": call_mids}).groupby("k", as_index=False).mean()
            feats.risk_neutral_density = risk_neutral_density_from_calls(
                spot, df["k"].tolist(), df["c"].tolist(), nearest_t
            )

    feats.meta = {
        "n_contracts": len(contracts),
        "nearest_expiration": nearest_exp_key,
        "nearest_t_years": None if nearest_t > 1e8 else nearest_t,
        "n_expiries": len(by_exp),
    }
    return feats
