"""Dealer Positioning feature engine — GEX, walls, and higher-order Greeks.

Dealer-style convention (customer long options / dealers short):
  exposure_i = greek_i * OI_i * multiplier
  signed: calls contribute +, puts contribute − for gamma-like quantities
  hedge requirement ≈ −Σ delta_i * OI_i * 100  (shares dealers must hold)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


MULTIPLIER = 100.0
R = 0.04  # risk-free default

# Locked Feature Engineering — Dealer Positioning field catalog
DEALER_POSITIONING_FIELDS: tuple[str, ...] = (
    "net_gex",
    "gamma_exposure",
    "gamma_by_strike",
    "gamma_flip",
    "zero_gamma",
    "call_wall",
    "put_wall",
    "dealer_hedge_requirement",
    "dealer_share_imbalance",
    "dealer_delta",
    "dealer_gamma",
    "dealer_charm",
    "dealer_vanna",
    "dealer_vomma",
    "dealer_vega",
    "dealer_theta",
    "dealer_speed",
    "dealer_color",
    "dealer_zomma",
    "dealer_ultima",
    "dealer_cross_gamma",
    "dealer_volga",
    "dealer_elasticity",
    "dealer_convexity",
    "dealer_liquidity_score",
    "dealer_inventory_estimate",
    "dealer_gamma_acceleration",
    "dealer_hedging_velocity",
)


@dataclass
class DealerPositioningFeatures:
    symbol: str
    as_of: str
    spot: float

    # Core GEX / walls
    net_gex: float = 0.0
    gamma_exposure: float = 0.0
    gamma_by_strike: list[dict[str, float]] = field(default_factory=list)
    gamma_flip: float | None = None
    zero_gamma: float | None = None
    call_wall: float | None = None
    put_wall: float | None = None

    # Hedge / imbalance
    dealer_hedge_requirement: float = 0.0  # shares
    dealer_share_imbalance: float = 0.0
    dealer_inventory_estimate: float = 0.0

    # First-order / common
    dealer_delta: float = 0.0
    dealer_gamma: float = 0.0
    dealer_vega: float = 0.0
    dealer_theta: float = 0.0

    # Higher-order
    dealer_charm: float = 0.0
    dealer_vanna: float = 0.0
    dealer_vomma: float = 0.0
    dealer_volga: float = 0.0  # alias of vomma
    dealer_speed: float = 0.0
    dealer_color: float = 0.0
    dealer_zomma: float = 0.0
    dealer_ultima: float = 0.0
    dealer_cross_gamma: float = 0.0
    dealer_elasticity: float = 0.0
    dealer_convexity: float = 0.0

    # Composite
    dealer_liquidity_score: float = 0.0
    dealer_gamma_acceleration: float = 0.0
    dealer_hedging_velocity: float = 0.0

    # Aux
    call_oi: float = 0.0
    put_oi: float = 0.0
    call_gex: float = 0.0
    put_gex: float = 0.0
    regime: str = "neutral_gamma"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat_features(self) -> dict[str, Any]:
        """Scalar feature row suitable for ML feature stores."""
        d = self.to_dict()
        d.pop("gamma_by_strike", None)
        d.pop("meta", None)
        return d


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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


def _normalize_iv(raw: Any) -> float:
    iv = float(raw or 0.35)
    if iv > 5:
        iv /= 100.0
    return max(min(iv, 3.0), 0.05)


def _t_years(contract: dict[str, Any], now: datetime) -> float:
    dte = contract.get("days_to_expiration")
    if dte is not None:
        try:
            return max(float(dte) / 365.0, 1.0 / 365.0)
        except (TypeError, ValueError):
            pass
    exp_str = contract.get("expiration_date")
    if not exp_str:
        return 30.0 / 365.0
    try:
        exp = datetime.fromisoformat(str(exp_str).replace("Z", "+00:00"))
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return max((exp - now).total_seconds() / (365.25 * 86400), 1.0 / 365.0)
    except ValueError:
        return 30.0 / 365.0


def _is_call(contract: dict[str, Any]) -> bool:
    pc = str(contract.get("put_call") or contract.get("side") or "call").lower()
    return pc.startswith("c")


def _d1_d2(spot: float, strike: float, t: float, iv: float, r: float = R) -> tuple[float, float]:
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    return d1, d2


def bs_greeks(
    spot: float,
    strike: float,
    t: float,
    iv: float,
    *,
    is_call: bool,
    r: float = R,
) -> dict[str, float]:
    """Black-Scholes greeks including higher-order terms."""
    if spot <= 0 or strike <= 0 or t <= 0 or iv <= 0:
        return {k: 0.0 for k in (
            "delta", "gamma", "vega", "theta", "charm", "vanna", "vomma",
            "speed", "color", "zomma", "ultima", "elasticity", "convexity",
        )}

    d1, d2 = _d1_d2(spot, strike, t, iv, r)
    pdf = _norm_pdf(d1)
    sqrt_t = math.sqrt(t)
    disc = math.exp(-r * t)

    gamma = pdf / (spot * iv * sqrt_t)
    vega = spot * pdf * sqrt_t / 100.0  # per 1 vol point
    # raw vega (not /100) for higher-order chain rules
    vega_raw = spot * pdf * sqrt_t

    if is_call:
        delta = _norm_cdf(d1)
        theta = (
            -(spot * pdf * iv) / (2 * sqrt_t)
            - r * strike * disc * _norm_cdf(d2)
        ) / 365.0
        elasticity = delta * spot / max(spot * delta, 1e-12)  # placeholder refined below
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (
            -(spot * pdf * iv) / (2 * sqrt_t)
            + r * strike * disc * _norm_cdf(-d2)
        ) / 365.0
        elasticity = delta * spot / max(abs(spot * delta), 1e-12)

    # Charm: dDelta/dTau (per day)
    charm = (
        -pdf * (2 * r * t - d2 * iv * sqrt_t) / (2 * t * iv * sqrt_t)
    ) / 365.0
    if not is_call:
        # put charm = call charm (same d1 path for digital part) — BS put charm equals call charm
        pass

    # Vanna: dDelta/dVol = dVega/dSpot
    vanna = -pdf * d2 / iv / 100.0

    # Vomma / Volga: dVega/dVol
    vomma = vega_raw * d1 * d2 / iv / 100.0  # per vol point on /100 vega scale-ish
    volga = vomma

    # Speed: dGamma/dSpot
    speed = -gamma / spot * (d1 / (iv * sqrt_t) + 1.0)

    # Color: dGamma/dTau (per day)
    color = (
        -pdf
        / (2 * spot * t * iv * sqrt_t)
        * (2 * r * t + 1.0 - d1 * ((2 * r * t - d2 * iv * sqrt_t) / (iv * sqrt_t)))
    ) / 365.0

    # Zomma: dGamma/dVol
    zomma = gamma * (d1 * d2 - 1.0) / iv / 100.0

    # Ultima: dVomma/dVol (approx)
    ultima = (-vega_raw / (iv * iv) * (d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2)) / 100.0

    # Elasticity (omega): delta * S / V — approximate with delta * S / (spot * max(|delta|,eps))
    # Better: use abs(delta)*S as notional sensitivity proxy when premium unknown
    premium_proxy = max(abs(delta) * spot * 0.01, 1e-6)
    elasticity = delta * spot / premium_proxy

    # Convexity proxy: gamma * S^2 (dollar gamma style unit)
    convexity = gamma * spot * spot

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "charm": float(charm),
        "vanna": float(vanna),
        "vomma": float(vomma),
        "volga": float(volga),
        "speed": float(speed),
        "color": float(color),
        "zomma": float(zomma),
        "ultima": float(ultima),
        "elasticity": float(elasticity),
        "convexity": float(convexity),
    }


def compute_dealer_positioning(
    matrix: dict[str, Any],
    *,
    as_of: str = "",
    prior: DealerPositioningFeatures | None = None,
    spot_bump_pct: float = 0.01,
) -> DealerPositioningFeatures:
    """Compute full dealer positioning feature block from an options matrix."""
    symbol = str(matrix.get("symbol", "")).upper()
    spot = float(matrix.get("underlying_price") or 0.0)
    contracts = _contracts(matrix)
    now = datetime.now(timezone.utc)
    as_of_date = as_of or str(matrix.get("as_of_date") or "")[:10]

    feats = DealerPositioningFeatures(symbol=symbol, as_of=as_of_date, spot=spot)
    if spot <= 0 or not contracts:
        feats.meta["error"] = "missing_spot_or_contracts"
        return feats

    levels: dict[float, float] = {}
    # Aggregators for dealer (short customer) exposures: −sign_callput * greek * OI * mult
    # For GEX we keep classic call+ / put− customer-long convention then flip for "dealer_*"

    call_gex = put_gex = 0.0
    tot_call_oi = tot_put_oi = 0.0
    tot_volume = 0.0

    agg = {
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0,
        "charm": 0.0,
        "vanna": 0.0,
        "vomma": 0.0,
        "speed": 0.0,
        "color": 0.0,
        "zomma": 0.0,
        "ultima": 0.0,
        "elasticity": 0.0,
        "convexity": 0.0,
        "cross_gamma": 0.0,
    }

    # For cross-gamma / hedge velocity: recompute net delta at bumped spot
    delta_at_spot = 0.0
    delta_at_up = 0.0

    for c in contracts:
        strike = float(c.get("strike_price") or 0)
        oi = float(c.get("open_interest") or 0)
        vol = float(c.get("total_volume") or 0)
        if strike <= 0 or oi <= 0:
            continue
        is_call = _is_call(c)
        sign = 1.0 if is_call else -1.0  # customer-long GEX sign
        t = _t_years(c, now)
        iv = _normalize_iv(c.get("volatility") or c.get("implied_volatility"))

        g = bs_greeks(spot, strike, t, iv, is_call=is_call)
        # Prefer quoted first-order greeks when present
        if c.get("gamma") not in (None, "", 0, 0.0):
            g["gamma"] = float(c["gamma"])
        if c.get("delta") not in (None, "", 0, 0.0):
            g["delta"] = float(c["delta"])
        if c.get("vega") not in (None, "", 0, 0.0):
            g["vega"] = float(c["vega"])
        if c.get("theta") not in (None, "", 0, 0.0):
            g["theta"] = float(c["theta"])

        gex_i = g["gamma"] * oi * MULTIPLIER * spot
        if is_call:
            call_gex += gex_i
            tot_call_oi += oi
            levels[strike] = levels.get(strike, 0.0) + gex_i
        else:
            put_gex += gex_i
            tot_put_oi += oi
            levels[strike] = levels.get(strike, 0.0) - gex_i

        tot_volume += vol
        w = oi * MULTIPLIER
        # Dealer short → opposite customer greek exposure
        dealer_sign = -1.0
        for key in (
            "delta", "gamma", "vega", "theta", "charm", "vanna", "vomma",
            "speed", "color", "zomma", "ultima", "elasticity", "convexity",
        ):
            # For delta: use raw signed delta; for gamma-like use GEX sign on customer then flip
            if key == "delta":
                agg[key] += dealer_sign * g[key] * w
            elif key in ("gamma", "speed", "color", "zomma", "convexity"):
                agg[key] += dealer_sign * sign * abs(g[key]) * w
            else:
                agg[key] += dealer_sign * g[key] * w

        # Cross gamma proxy: gamma contribution * relative strike distance
        moneyness = (strike - spot) / spot
        agg["cross_gamma"] += dealer_sign * sign * g["gamma"] * w * moneyness

        delta_at_spot += dealer_sign * g["delta"] * w
        g_up = bs_greeks(spot * (1.0 + spot_bump_pct), strike, t, iv, is_call=is_call)
        if c.get("delta") not in (None, "", 0, 0.0) and abs(spot_bump_pct) < 1e-12:
            pass
        delta_at_up += dealer_sign * g_up["delta"] * w

    net_gex = call_gex - put_gex
    feats.net_gex = float(net_gex)
    feats.gamma_exposure = float(abs(net_gex))
    feats.call_gex = float(call_gex)
    feats.put_gex = float(put_gex)
    feats.call_oi = float(tot_call_oi)
    feats.put_oi = float(tot_put_oi)

    # Gamma by strike (sorted)
    feats.gamma_by_strike = [
        {"strike": float(k), "gex": float(v)}
        for k, v in sorted(levels.items(), key=lambda kv: kv[0])
    ]

    # Walls / flip / zero gamma
    if levels:
        feats.call_wall = float(max(levels, key=levels.get))
        feats.put_wall = float(min(levels, key=levels.get))
        strikes = sorted(levels.keys())
        flip = None
        for i in range(1, len(strikes)):
            a, b = levels[strikes[i - 1]], levels[strikes[i]]
            if a == 0:
                flip = strikes[i - 1]
                break
            if a * b < 0:
                # linear interpolate zero crossing
                w = abs(a) / (abs(a) + abs(b))
                flip = strikes[i - 1] * (1 - w) + strikes[i] * w
                break
        feats.gamma_flip = float(flip) if flip is not None else None
        feats.zero_gamma = feats.gamma_flip

    if net_gex > 0:
        feats.regime = "positive_gamma"
    elif net_gex < 0:
        feats.regime = "negative_gamma"
    else:
        feats.regime = "neutral_gamma"

    # Dealer aggregates
    feats.dealer_delta = float(agg["delta"])
    feats.dealer_gamma = float(agg["gamma"])
    feats.dealer_vega = float(agg["vega"])
    feats.dealer_theta = float(agg["theta"])
    feats.dealer_charm = float(agg["charm"])
    feats.dealer_vanna = float(agg["vanna"])
    feats.dealer_vomma = float(agg["vomma"])
    feats.dealer_volga = float(agg["vomma"])
    feats.dealer_speed = float(agg["speed"])
    feats.dealer_color = float(agg["color"])
    feats.dealer_zomma = float(agg["zomma"])
    feats.dealer_ultima = float(agg["ultima"])
    feats.dealer_cross_gamma = float(agg["cross_gamma"])
    feats.dealer_elasticity = float(agg["elasticity"])
    feats.dealer_convexity = float(agg["convexity"])

    # Hedge requirement = −customer delta shares = dealer delta (already dealer-signed)
    feats.dealer_hedge_requirement = float(delta_at_spot)
    # Share imbalance: call vs put OI skew in share-equivalent gamma units
    total_oi = tot_call_oi + tot_put_oi
    feats.dealer_share_imbalance = float(
        (tot_call_oi - tot_put_oi) / total_oi if total_oi > 0 else 0.0
    )
    # Inventory estimate: −net delta shares (dealers inventory to hedge)
    feats.dealer_inventory_estimate = float(-delta_at_spot)

    # Liquidity score: volume / OI concentration (higher = easier to hedge)
    if total_oi > 0:
        feats.dealer_liquidity_score = float(
            max(0.0, min(1.0, (tot_volume / total_oi) / 2.0))
        )
    else:
        feats.dealer_liquidity_score = 0.0

    # Gamma acceleration: change in dealer gamma vs prior, else speed * spot move proxy
    if prior is not None:
        feats.dealer_gamma_acceleration = float(feats.dealer_gamma - prior.dealer_gamma)
        feats.dealer_hedging_velocity = float(
            feats.dealer_hedge_requirement - prior.dealer_hedge_requirement
        )
    else:
        # Instantaneous: d(hedge)/dS ≈ gamma exposure in share terms
        d_delta = delta_at_up - delta_at_spot
        feats.dealer_hedging_velocity = float(d_delta / max(spot_bump_pct * spot, 1e-9))
        feats.dealer_gamma_acceleration = float(feats.dealer_speed * spot)

    feats.meta = {
        "n_contracts": len(contracts),
        "multiplier": MULTIPLIER,
        "spot_bump_pct": spot_bump_pct,
        "convention": "dealer_short_customer_long",
    }
    return feats


def dealer_feature_row(matrix: dict[str, Any], *, as_of: str = "") -> dict[str, Any]:
    """Convenience: flat ML-ready dealer positioning features."""
    return compute_dealer_positioning(matrix, as_of=as_of).flat_features()
