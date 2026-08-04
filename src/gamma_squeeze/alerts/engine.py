"""Alert Engine — detect squeeze / dealer / wall / IV / pattern / earnings events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from gamma_squeeze.alerts.channels import CHANNELS, channel_config
from gamma_squeeze.config import alpaca_configured, resolve_matrix_root
from gamma_squeeze.features.feature_store import build_features_for_symbol, load_features
from gamma_squeeze.ingest.alpaca_backup_client import list_local_dates, load_local_matrix
from gamma_squeeze.ingest.macro_client import fetch_calendar_near
from gamma_squeeze.serve.export_forecasts import load_latest_forecast


ALERT_TRIGGERS: tuple[str, ...] = (
    "Gamma Squeeze Probability",
    "Dealer Flip",
    "Gamma Ramp",
    "Call Wall Break",
    "Put Wall Break",
    "Large IV Expansion",
    "Large Dealer Hedge",
    "Pattern Breakout",
    "Earnings Risk",
)

BREAKOUT_PATTERNS = {
    "range_breakout",
    "Bull Flag",
    "Ascending Triangle",
    "Cup Handle",
    "Double Bottom",
    "Inverse Head Shoulders",
    "Pennant",
    "Channel",
    "Volatility Squeeze",
}


@dataclass
class AlertThresholds:
    min_probability: float = 0.55
    min_confidence: float = 0.45
    horizon_days: int = 5
    dealer_flip_distance_pct: float = 0.015  # within 1.5% of flip
    gamma_ramp_abs: float = 2.5e4
    wall_break_buffer_pct: float = 0.002  # 0.2% through wall
    iv_expansion_ratio: float = 1.25  # atm_iv / rvol
    iv_expansion_abs: float = 0.08
    dealer_hedge_shares: float = 5.0e4
    pattern_min_confidence: float = 0.55
    earnings_look_ahead_days: int = 5

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _f(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return default


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _alert(
    *,
    type_: str,
    severity: str,
    symbol: str,
    message: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": type_,
        "severity": severity,
        "symbol": symbol,
        "message": message,
        "meta": meta or {},
        "ts": _ts(),
        "data_plane": "alpaca",
    }


def _load_matrix(symbol: str, as_of: str | None) -> tuple[dict[str, Any] | None, str]:
    if alpaca_configured():
        try:
            from gamma_squeeze.ingest.alpaca_api import fetch_live_options_matrix

            live = fetch_live_options_matrix(symbol, as_of=as_of)
            if live.get("contracts"):
                return live, "alpaca_api"
        except Exception:  # noqa: BLE001
            pass
    root = resolve_matrix_root()
    if as_of:
        local = load_local_matrix(root, symbol, as_of)
        if local:
            return local, "ssd"
    dates = list_local_dates(root, symbol) or []
    if dates:
        local = load_local_matrix(root, symbol, dates[-1])
        if local:
            return local, "ssd"
    return None, "none"


def evaluate_alert_engine(
    symbol: str,
    *,
    thresholds: AlertThresholds | None = None,
    notify: bool = False,
    channels: list[str] | None = None,
    use_live_alpaca: bool = True,
) -> dict[str, Any]:
    """Run all alert triggers using Alpaca-forward market/options data."""
    from gamma_squeeze.alerts.channels import dispatch_alerts

    th = thresholds or AlertThresholds()
    sym = symbol.upper()
    feat = load_features(sym)
    if feat.empty:
        feat = build_features_for_symbol(
            sym,
            include_skew=False,
            include_macro=True,
            include_technicals=True,
            include_dealer=True,
            include_options_metrics=True,
        )
    if feat.empty:
        return {
            "symbol": sym,
            "error": "no_features",
            "fired": False,
            "alerts": [],
            "thresholds": th.to_dict(),
            "alpaca_configured": alpaca_configured(),
        }

    row = feat.sort_values("as_of").iloc[-1].to_dict()
    as_of = str(row.get("as_of"))[:10]
    spot = _f(row, "spot", default=0.0)

    forecast = load_latest_forecast(sym)
    squeeze_payload: dict[str, Any] = {}
    has_composite = bool((forecast or {}).get("meta", {}).get("composite"))
    if (not forecast) or (not has_composite):
        try:
            from services.gamma_squeeze_engine.service import run_squeeze

            squeeze_payload = run_squeeze(sym, persist=False, composite=True) or {}
            if not forecast:
                forecast = squeeze_payload.get("forecast") or squeeze_payload
        except Exception:  # noqa: BLE001
            if not forecast:
                forecast = None

    matrix = None
    origin = "features"
    if use_live_alpaca:
        matrix, origin = _load_matrix(sym, as_of)

    dealer: dict[str, Any] = {}
    if matrix:
        try:
            from gamma_squeeze.dealer.hedge_demand import simulate_dealer_hedging

            dealer = simulate_dealer_hedging(matrix, as_of=as_of).to_dict()
            if not spot:
                spot = float(dealer.get("spot") or matrix.get("underlying_price") or 0)
        except Exception:  # noqa: BLE001
            dealer = {}

    patterns: dict[str, Any] = {}
    try:
        from services.pattern_recognition.service import run_patterns

        patterns = run_patterns(sym) or {}
    except Exception:  # noqa: BLE001
        patterns = {}

    alerts: list[dict[str, Any]] = []

    # 1) Gamma Squeeze Probability (configurable threshold)
    horizons = (forecast or {}).get("horizons") or []
    # horizons may be list of dicts or HorizonForecast-like
    def _hz_days(x: Any) -> int | None:
        if isinstance(x, dict):
            return x.get("horizon_days")
        return getattr(x, "horizon_days", None)

    def _hz_get(x: Any, key: str, default: float = 0.0) -> float:
        if isinstance(x, dict):
            return float(x.get(key) or default)
        return float(getattr(x, key, default) or default)

    h = next((x for x in horizons if _hz_days(x) == th.horizon_days), None)
    if h is None and horizons:
        h = horizons[min(th.horizon_days - 1, len(horizons) - 1)]
    # composite overlay — top-level squeeze payload, forecast.meta.composite, or horizon
    fo = (squeeze_payload.get("final_output") or {}) if squeeze_payload else {}
    comp = (forecast or {}).get("meta", {}).get("composite") or {}
    p = float(
        fo.get("probability")
        or squeeze_payload.get("probability")
        or comp.get("probability")
        or (_hz_get(h, "squeeze_probability") if h is not None else 0)
        or 0
    )
    c = float(
        fo.get("confidence")
        or squeeze_payload.get("confidence")
        or comp.get("confidence")
        or (_hz_get(h, "confidence_score") if h is not None else 0)
        or 0
    )
    if p >= th.min_probability:
        sev = "high" if p >= 0.7 else "medium"
        if c < th.min_confidence:
            sev = "medium" if sev == "high" else "low"
        alerts.append(
            _alert(
                type_="gamma_squeeze_probability",
                severity=sev,
                symbol=sym,
                message=(
                    f"{sym}: squeeze P={p:.2f} (threshold {th.min_probability:.2f}), "
                    f"confidence={c:.2f} @ {th.horizon_days}d"
                ),
                meta={"probability": p, "confidence": c, "horizon_days": th.horizon_days},
            )
        )

    # 2) Dealer Flip
    flip = (dealer.get("dealer_position_flip") or {}) if dealer else {}
    flip_spot = flip.get("gamma_flip_spot") or _f(row, "gamma_flip", "zero_gamma", default=0.0)
    if flip_spot and spot > 0:
        dist = abs(float(flip_spot) - spot) / spot
        if dist <= th.dealer_flip_distance_pct:
            alerts.append(
                _alert(
                    type_="dealer_flip",
                    severity="high",
                    symbol=sym,
                    message=(
                        f"{sym}: dealer flip near spot — flip={float(flip_spot):.2f}, "
                        f"spot={spot:.2f}, dist={dist:.2%}"
                    ),
                    meta={"flip_spot": float(flip_spot), "spot": spot, "distance_pct": dist},
                )
            )

    # 3) Gamma Ramp
    ramp = float(dealer.get("gamma_ramp") or 0.0)
    if abs(ramp) >= th.gamma_ramp_abs:
        alerts.append(
            _alert(
                type_="gamma_ramp",
                severity="high" if abs(ramp) >= 2 * th.gamma_ramp_abs else "medium",
                symbol=sym,
                message=f"{sym}: gamma ramp forming — |ramp|={abs(ramp):.3g} (threshold {th.gamma_ramp_abs:.3g})",
                meta={"gamma_ramp": ramp},
            )
        )

    # 4/5) Call / Put wall breaks
    call_wall = _f(row, "call_wall", default=0.0) or float(
        ((dealer.get("meta") or {}).get("positioning") or {}).get("call_wall") or 0
    )
    put_wall = _f(row, "put_wall", default=0.0) or float(
        ((dealer.get("meta") or {}).get("positioning") or {}).get("put_wall") or 0
    )
    if call_wall and spot > 0 and spot >= call_wall * (1.0 - th.wall_break_buffer_pct):
        alerts.append(
            _alert(
                type_="call_wall_break",
                severity="high",
                symbol=sym,
                message=f"{sym}: call wall break/test — spot={spot:.2f} vs call_wall={call_wall:.2f}",
                meta={"spot": spot, "call_wall": call_wall},
            )
        )
    if put_wall and spot > 0 and spot <= put_wall * (1.0 + th.wall_break_buffer_pct):
        alerts.append(
            _alert(
                type_="put_wall_break",
                severity="high",
                symbol=sym,
                message=f"{sym}: put wall break/test — spot={spot:.2f} vs put_wall={put_wall:.2f}",
                meta={"spot": spot, "put_wall": put_wall},
            )
        )

    # 6) Large IV Expansion
    atm_iv = _f(row, "atm_iv", "iv", default=0.0)
    rvol = _f(row, "rvol_10d", default=0.0)
    iv_ratio = (atm_iv / rvol) if rvol > 1e-6 else 0.0
    if atm_iv >= th.iv_expansion_abs and (iv_ratio >= th.iv_expansion_ratio or atm_iv - rvol >= 0.08):
        alerts.append(
            _alert(
                type_="large_iv_expansion",
                severity="medium",
                symbol=sym,
                message=f"{sym}: large IV expansion — ATM IV={atm_iv:.2%} vs RVol={rvol:.2%} (ratio={iv_ratio:.2f})",
                meta={"atm_iv": atm_iv, "rvol_10d": rvol, "ratio": iv_ratio},
            )
        )

    # 7) Large Dealer Hedge
    hedge = abs(float(dealer.get("delta_hedging") or _f(row, "dealer_hedge_requirement")))
    ehv = dealer.get("expected_hedging_volume") or {}
    hedge_vol = abs(float(ehv.get("shares") or ehv.get("expected_shares") or 0))
    hedge_signal = max(hedge, hedge_vol)
    if hedge_signal >= th.dealer_hedge_shares:
        alerts.append(
            _alert(
                type_="large_dealer_hedge",
                severity="high" if hedge_signal >= 2 * th.dealer_hedge_shares else "medium",
                symbol=sym,
                message=f"{sym}: large dealer hedge — |hedge|≈{hedge_signal:.3g} shares",
                meta={"delta_hedging": hedge, "expected_hedging_shares": hedge_vol},
            )
        )

    # 8) Pattern Breakout
    pats = patterns.get("patterns") or patterns.get("top") or []
    for pat in pats:
        name = str(pat.get("pattern") or "")
        conf = float(pat.get("confidence") or pat.get("probability") or 0)
        if name in BREAKOUT_PATTERNS and conf >= th.pattern_min_confidence:
            alerts.append(
                _alert(
                    type_="pattern_breakout",
                    severity="medium",
                    symbol=sym,
                    message=f"{sym}: pattern breakout — {name} (confidence={conf:.2f})",
                    meta={"pattern": name, "confidence": conf},
                )
            )
            break

    # 9) Earnings Risk
    earnings_hits: list[dict[str, Any]] = []
    try:
        events = fetch_calendar_near(as_of, look_ahead_days=th.earnings_look_ahead_days)
        for e in events:
            title = str(e.get("title") or e.get("event") or e.get("name") or "").lower()
            if "earn" in title or str(e.get("type") or "").lower() == "earnings":
                earnings_hits.append(e)
        # also corporate earnings adapter best-effort
        if not earnings_hits:
            try:
                from gamma_squeeze.data_sources.registry import get_adapter, load_all_adapters

                load_all_adapters()
                out = get_adapter("corporate.earnings").fetch(sym)
                records = out.get("records") or (out.get("data") or {}).get("earnings") or []
                for r in records[:10]:
                    d = str(r.get("date") or r.get("reportDate") or "")[:10]
                    if d and as_of <= d:
                        earnings_hits.append(r)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    if earnings_hits:
        nxt = earnings_hits[0]
        alerts.append(
            _alert(
                type_="earnings_risk",
                severity="medium",
                symbol=sym,
                message=(
                    f"{sym}: earnings risk — upcoming event "
                    f"{nxt.get('date') or nxt.get('reportDate') or nxt.get('title')}"
                ),
                meta={"event": nxt, "n_events": len(earnings_hits)},
            )
        )

    fired = bool(alerts)
    payload = {
        "symbol": sym,
        "as_of": as_of,
        "fired": fired,
        "alerts": alerts,
        "n_alerts": len(alerts),
        "thresholds": th.to_dict(),
        "triggers": list(ALERT_TRIGGERS),
        "channels": list(CHANNELS),
        "channel_status": channel_config(),
        "data_origin": origin,
        "alpaca_configured": alpaca_configured(),
        "spot": spot,
        "notifications": [],
    }
    if notify and fired:
        payload["notifications"] = dispatch_alerts(sym, payload, channels=channels)
    return payload


def engine_meta() -> dict[str, Any]:
    from gamma_squeeze.ingest.alpaca_api import alpaca_api_health

    return {
        "service": "trade_alerts",
        "engine": "alert-engine-v1",
        "notify_when": list(ALERT_TRIGGERS),
        "support": list(CHANNELS),
        "default_thresholds": AlertThresholds().to_dict(),
        "configurable": [
            "min_probability",
            "min_confidence",
            "horizon_days",
            "dealer_flip_distance_pct",
            "gamma_ramp_abs",
            "wall_break_buffer_pct",
            "iv_expansion_ratio",
            "iv_expansion_abs",
            "dealer_hedge_shares",
            "pattern_min_confidence",
            "earnings_look_ahead_days",
        ],
        "data_plane": "alpaca",
        "alpaca": alpaca_api_health(),
        "channel_status": channel_config(),
        "endpoints": [
            "/v1/meta",
            "/v1/alerts",
            "/v1/alerts/{symbol}",
            "/v1/notify",
            "/v1/ws/alerts",
        ],
    }
