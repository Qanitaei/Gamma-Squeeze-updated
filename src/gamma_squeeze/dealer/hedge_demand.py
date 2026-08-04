"""Dealer hedging simulation — Δ-hedge demand, flow map, inventory & exhaustion."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from gamma_squeeze.serve.schemas import HedgeDemandPoint

try:
    from gamma_squeeze.features.dealer_positioning import compute_dealer_positioning
except Exception:  # pragma: no cover
    compute_dealer_positioning = None  # type: ignore


DEALER_ESTIMATES = (
    "dealer_share_purchases",
    "dealer_share_sales",
    "gamma_ramp",
    "strike_migration",
    "delta_hedging",
    "dynamic_gamma",
    "dealer_inventory",
    "dealer_liquidity",
    "dealer_exhaustion",
    "dealer_position_flip",
)

DEALER_ESTIMATE_LABELS = {
    "dealer_share_purchases": "Dealer Share Purchases",
    "dealer_share_sales": "Dealer Share Sales",
    "gamma_ramp": "Gamma Ramp",
    "strike_migration": "Strike Migration",
    "delta_hedging": "Delta Hedging",
    "dynamic_gamma": "Dynamic Gamma",
    "dealer_inventory": "Dealer Inventory",
    "dealer_liquidity": "Dealer Liquidity",
    "dealer_exhaustion": "Dealer Exhaustion",
    "dealer_position_flip": "Dealer Position Flip",
}

DEALER_OUTPUTS = (
    "dealer_flow_map",
    "dealer_demand_curve",
    "expected_hedging_volume",
)

DEALER_OUTPUT_LABELS = {
    "dealer_flow_map": "Dealer Flow Map",
    "dealer_demand_curve": "Dealer Demand Curve",
    "expected_hedging_volume": "Expected Hedging Volume",
}


@dataclass
class DealerHedgingSimulation:
    symbol: str
    as_of: str
    spot: float

    # Estimates
    dealer_share_purchases: float = 0.0
    dealer_share_sales: float = 0.0
    gamma_ramp: float = 0.0
    strike_migration: dict[str, Any] = field(default_factory=dict)
    delta_hedging: float = 0.0
    dynamic_gamma: list[dict[str, float]] = field(default_factory=list)
    dealer_inventory: float = 0.0
    dealer_liquidity: float = 0.0
    dealer_exhaustion: float = 0.0
    dealer_position_flip: dict[str, Any] = field(default_factory=dict)

    # Outputs
    dealer_flow_map: list[dict[str, Any]] = field(default_factory=list)
    dealer_demand_curve: list[dict[str, Any]] = field(default_factory=list)
    expected_hedging_volume: dict[str, Any] = field(default_factory=dict)

    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["estimates"] = list(DEALER_ESTIMATES)
        d["estimate_labels"] = dict(DEALER_ESTIMATE_LABELS)
        d["outputs"] = list(DEALER_OUTPUTS)
        d["output_labels"] = dict(DEALER_OUTPUT_LABELS)
        return d


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


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _bs_gamma(spot: float, strike: float, t_years: float, iv: float = 0.35, r: float = 0.04) -> float:
    if t_years <= 0 or spot <= 0 or strike <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    return _norm_pdf(d1) / (spot * iv * math.sqrt(t_years))


def _spot0(matrix: dict[str, Any]) -> float:
    return float(matrix.get("underlying_price") or matrix.get("spot") or 0.0)


def compute_hedge_demand_curve(
    matrix: dict[str, Any],
    *,
    grid_pct: float = 0.08,
    n_points: int = 17,
) -> list[HedgeDemandPoint]:
    """Estimate dealer share/dollar delta hedge demand vs spot.

    Convention: dealers short retail options → hedge demand ≈ −Σ (delta_i * OI_i * 100)
    evaluated with BS greeks re-priced on a spot grid. Gamma exposure at each spot
    approximates local hedging intensity.
    """
    spot0 = _spot0(matrix)
    contracts = _contracts(matrix)
    if spot0 <= 0 or not contracts:
        return []

    lo = spot0 * (1.0 - grid_pct)
    hi = spot0 * (1.0 + grid_pct)
    if n_points < 3:
        n_points = 3
    step = (hi - lo) / (n_points - 1)

    points: list[HedgeDemandPoint] = []
    for i in range(n_points):
        spot = lo + step * i
        delta_shares = 0.0
        gamma_exp = 0.0
        for c in contracts:
            strike = float(c.get("strike_price") or 0)
            oi = float(c.get("open_interest") or 0)
            if strike <= 0 or oi <= 0:
                continue
            pc = str(c.get("put_call") or c.get("side") or "").lower()
            is_call = pc.startswith("c") or pc == "call"
            iv = float(c.get("volatility") or c.get("implied_volatility") or 0.35)
            if iv > 5:
                iv /= 100.0
            iv = max(min(iv, 3.0), 0.05)
            dte = float(c.get("days_to_expiration") or 30)
            t_years = max(dte / 365.0, 1 / 365)
            gamma = float(c.get("gamma") or 0) or _bs_gamma(spot, strike, t_years, iv)
            delta0 = float(c.get("delta") or 0)
            if not delta0:
                delta0 = 0.5 if is_call else -0.5
            if not is_call and delta0 > 0:
                delta0 = -abs(delta0)
            local_delta = delta0 + gamma * (spot - spot0)
            sign = -1.0
            delta_shares += sign * local_delta * oi * 100.0
            g_sign = 1.0 if is_call else -1.0
            gamma_exp += g_sign * gamma * oi * 100.0 * spot

        points.append(
            HedgeDemandPoint(
                spot=float(round(spot, 4)),
                delta_shares=float(delta_shares),
                delta_dollars=float(delta_shares * spot),
                gamma_exposure=float(gamma_exp),
            )
        )
    return points


def _gamma_by_strike_at_spot(matrix: dict[str, Any], spot: float) -> list[dict[str, float]]:
    """Signed GEX by strike at a hypothetical spot (for strike migration)."""
    contracts = _contracts(matrix)
    by_strike: dict[float, float] = {}
    for c in contracts:
        strike = float(c.get("strike_price") or 0)
        oi = float(c.get("open_interest") or 0)
        if strike <= 0 or oi <= 0:
            continue
        pc = str(c.get("put_call") or c.get("side") or "").lower()
        is_call = pc.startswith("c") or pc == "call"
        iv = float(c.get("volatility") or c.get("implied_volatility") or 0.35)
        if iv > 5:
            iv /= 100.0
        iv = max(min(iv, 3.0), 0.05)
        dte = float(c.get("days_to_expiration") or 30)
        t_years = max(dte / 365.0, 1 / 365)
        gamma = float(c.get("gamma") or 0) or _bs_gamma(spot, strike, t_years, iv)
        g_sign = 1.0 if is_call else -1.0
        by_strike[strike] = by_strike.get(strike, 0.0) + g_sign * gamma * oi * 100.0 * spot
    return [{"strike": k, "gex": v} for k, v in sorted(by_strike.items())]


def _dominant_strike(gbs: list[dict[str, float]], *, side: str) -> float | None:
    if not gbs:
        return None
    if side == "call":
        row = max(gbs, key=lambda r: r["gex"])
        return float(row["strike"]) if row["gex"] > 0 else None
    row = min(gbs, key=lambda r: r["gex"])
    return float(row["strike"]) if row["gex"] < 0 else None


def _liquidity_score(matrix: dict[str, Any]) -> float:
    contracts = _contracts(matrix)
    if not contracts:
        return 0.0
    vol = sum(float(c.get("total_volume") or c.get("volume") or 0) for c in contracts)
    oi = sum(float(c.get("open_interest") or 0) for c in contracts)
    if oi <= 0:
        return 0.0
    # higher turnover → more liquid hedging
    turnover = vol / oi
    return float(max(0.0, min(1.0, math.tanh(turnover * 2.0))))


def simulate_dealer_hedging(
    matrix: dict[str, Any],
    *,
    grid_pct: float = 0.08,
    n_points: int = 17,
    expected_move_pct: float | None = None,
    as_of: str | None = None,
) -> DealerHedgingSimulation:
    """
    Full dealer hedging simulation.

    Estimates share purchases/sales, gamma ramp, strike migration, delta hedging,
    dynamic gamma, inventory, liquidity, exhaustion, and position flip.
    Outputs flow map, demand curve, and expected hedging volume.
    """
    spot0 = _spot0(matrix)
    symbol = str(matrix.get("symbol") or "").upper()
    date = as_of or str(matrix.get("as_of_date") or matrix.get("as_of") or "")
    sim = DealerHedgingSimulation(symbol=symbol, as_of=date, spot=spot0)
    if spot0 <= 0:
        sim.meta["error"] = "missing_spot"
        return sim

    curve = compute_hedge_demand_curve(matrix, grid_pct=grid_pct, n_points=n_points)
    if not curve:
        sim.meta["error"] = "empty_curve"
        return sim

    # Demand curve output
    sim.dealer_demand_curve = [asdict(p) for p in curve]

    # Find spot-nearest point
    mid_i = min(range(len(curve)), key=lambda i: abs(curve[i].spot - spot0))
    mid = curve[mid_i]
    sim.delta_hedging = float(mid.delta_shares)
    sim.dealer_inventory = float(-mid.delta_shares)  # inventory ≈ opposite of hedge book

    # Dynamic gamma path
    sim.dynamic_gamma = [
        {"spot": float(p.spot), "gamma_exposure": float(p.gamma_exposure)} for p in curve
    ]

    # Gamma ramp: dGEX/dS around spot (central difference)
    if 0 < mid_i < len(curve) - 1:
        d_spot = curve[mid_i + 1].spot - curve[mid_i - 1].spot
        d_gex = curve[mid_i + 1].gamma_exposure - curve[mid_i - 1].gamma_exposure
        sim.gamma_ramp = float(d_gex / d_spot) if abs(d_spot) > 1e-9 else 0.0
    elif mid_i == 0 and len(curve) > 1:
        d_spot = curve[1].spot - curve[0].spot
        d_gex = curve[1].gamma_exposure - curve[0].gamma_exposure
        sim.gamma_ramp = float(d_gex / d_spot) if abs(d_spot) > 1e-9 else 0.0

    # Flow map: incremental hedge as spot walks the grid
    flow: list[dict[str, Any]] = []
    purchases = 0.0
    sales = 0.0
    prev_shares = curve[0].delta_shares
    for i, p in enumerate(curve):
        if i == 0:
            delta_flow = 0.0
        else:
            delta_flow = p.delta_shares - prev_shares
        buy = max(delta_flow, 0.0)
        sell = max(-delta_flow, 0.0)
        purchases += buy
        sales += sell
        flow.append(
            {
                "spot": float(p.spot),
                "dealer_share_purchases": float(buy),
                "dealer_share_sales": float(sell),
                "net_flow": float(delta_flow),
                "cumulative_hedge_shares": float(p.delta_shares),
                "gamma_exposure": float(p.gamma_exposure),
                "delta_dollars": float(p.delta_dollars),
            }
        )
        prev_shares = p.delta_shares
    sim.dealer_flow_map = flow
    sim.dealer_share_purchases = float(purchases)
    sim.dealer_share_sales = float(sales)

    # Strike migration: dominant call/put walls at low / mid / high spot
    lo_spot, hi_spot = curve[0].spot, curve[-1].spot
    g_lo = _gamma_by_strike_at_spot(matrix, lo_spot)
    g_mid = _gamma_by_strike_at_spot(matrix, spot0)
    g_hi = _gamma_by_strike_at_spot(matrix, hi_spot)
    sim.strike_migration = {
        "low_spot": {
            "spot": lo_spot,
            "call_wall": _dominant_strike(g_lo, side="call"),
            "put_wall": _dominant_strike(g_lo, side="put"),
        },
        "spot": {
            "spot": spot0,
            "call_wall": _dominant_strike(g_mid, side="call"),
            "put_wall": _dominant_strike(g_mid, side="put"),
        },
        "high_spot": {
            "spot": hi_spot,
            "call_wall": _dominant_strike(g_hi, side="call"),
            "put_wall": _dominant_strike(g_hi, side="put"),
        },
        "call_wall_shift": None,
        "put_wall_shift": None,
    }
    cw0 = sim.strike_migration["spot"]["call_wall"]
    cw1 = sim.strike_migration["high_spot"]["call_wall"]
    pw0 = sim.strike_migration["spot"]["put_wall"]
    pw1 = sim.strike_migration["low_spot"]["put_wall"]
    if cw0 is not None and cw1 is not None:
        sim.strike_migration["call_wall_shift"] = float(cw1 - cw0)
    if pw0 is not None and pw1 is not None:
        sim.strike_migration["put_wall_shift"] = float(pw1 - pw0)

    # Position flip: where hedge demand or GEX crosses zero on the grid
    flip_spot = None
    flip_kind = None
    for a, b in zip(curve[:-1], curve[1:]):
        if a.gamma_exposure == 0:
            flip_spot, flip_kind = a.spot, "gamma"
            break
        if a.gamma_exposure * b.gamma_exposure < 0:
            # linear interpolate
            t = abs(a.gamma_exposure) / (abs(a.gamma_exposure) + abs(b.gamma_exposure) + 1e-12)
            flip_spot = a.spot + t * (b.spot - a.spot)
            flip_kind = "gamma"
            break
    hedge_flip = None
    for a, b in zip(curve[:-1], curve[1:]):
        if a.delta_shares * b.delta_shares < 0:
            t = abs(a.delta_shares) / (abs(a.delta_shares) + abs(b.delta_shares) + 1e-12)
            hedge_flip = a.spot + t * (b.spot - a.spot)
            break
    sim.dealer_position_flip = {
        "gamma_flip_spot": float(flip_spot) if flip_spot is not None else None,
        "hedge_flip_spot": float(hedge_flip) if hedge_flip is not None else None,
        "kind": flip_kind,
        "distance_pct": float((flip_spot - spot0) / spot0) if flip_spot is not None else None,
        "below_spot": bool(flip_spot is not None and flip_spot < spot0),
    }

    # Liquidity + exhaustion
    liq = _liquidity_score(matrix)
    sim.dealer_liquidity = liq
    # Capacity proxy: liquid fraction of total OI * 100 shares
    contracts = _contracts(matrix)
    total_oi = sum(float(c.get("open_interest") or 0) for c in contracts)
    capacity = max(total_oi * 100.0 * max(liq, 0.05), 1.0)
    # Exhaustion from peak |incremental| flow vs capacity and |current hedge|
    peak_flow = max((abs(f["net_flow"]) for f in flow), default=0.0)
    sim.dealer_exhaustion = float(
        max(0.0, min(1.0, 0.5 * (peak_flow / capacity) + 0.5 * (abs(sim.delta_hedging) / capacity)))
    )

    # Enrich from dealer positioning block when available
    if compute_dealer_positioning is not None:
        try:
            pos = compute_dealer_positioning(matrix, as_of=date or None)
            if pos.dealer_liquidity_score:
                sim.dealer_liquidity = float(0.5 * sim.dealer_liquidity + 0.5 * pos.dealer_liquidity_score)
            if pos.dealer_hedge_requirement:
                # blend curve mid with positioning hedge requirement
                sim.delta_hedging = float(0.5 * sim.delta_hedging + 0.5 * pos.dealer_hedge_requirement)
                sim.dealer_inventory = float(-sim.delta_hedging)
            sim.meta["positioning"] = {
                "net_gex": pos.net_gex,
                "gamma_flip": pos.gamma_flip,
                "call_wall": pos.call_wall,
                "put_wall": pos.put_wall,
                "regime": pos.regime,
            }
        except Exception as exc:  # noqa: BLE001
            sim.meta["positioning_error"] = str(exc)

    # Expected hedging volume over ± expected move
    em = expected_move_pct
    if em is None:
        # crude: 1σ from ATM IV if present else grid/2
        ivs = [
            float(c.get("volatility") or c.get("implied_volatility") or 0)
            for c in contracts
            if c.get("volatility") or c.get("implied_volatility")
        ]
        ivs = [v / 100.0 if v > 5 else v for v in ivs if v]
        atm_iv = float(sorted(ivs)[len(ivs) // 2]) if ivs else 0.25
        em = atm_iv * math.sqrt(1 / 252.0)  # 1-day
    band_lo = spot0 * (1.0 - abs(em))
    band_hi = spot0 * (1.0 + abs(em))
    in_band = [p for p in curve if band_lo <= p.spot <= band_hi]
    if len(in_band) >= 2:
        vol_shares = abs(in_band[-1].delta_shares - in_band[0].delta_shares)
    else:
        # scale full-grid purchases+sales by em/grid
        vol_shares = (purchases + sales) * min(1.0, abs(em) / max(grid_pct, 1e-9))
    sim.expected_hedging_volume = {
        "shares": float(vol_shares),
        "dollars": float(vol_shares * spot0),
        "expected_move_pct": float(em),
        "band": {"low": float(band_lo), "high": float(band_hi)},
        "purchases_in_band": float(
            sum(f["dealer_share_purchases"] for f in flow if band_lo <= f["spot"] <= band_hi)
        ),
        "sales_in_band": float(
            sum(f["dealer_share_sales"] for f in flow if band_lo <= f["spot"] <= band_hi)
        ),
    }

    sim.meta.update(
        {
            "grid_pct": grid_pct,
            "n_points": n_points,
            "convention": "dealers_short_customer_long",
            "n_contracts": len(contracts),
        }
    )
    return sim
