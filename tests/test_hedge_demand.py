from gamma_squeeze.dealer.hedge_demand import (
    DEALER_ESTIMATE_LABELS,
    DEALER_ESTIMATES,
    DEALER_OUTPUT_LABELS,
    DEALER_OUTPUTS,
    compute_hedge_demand_curve,
    simulate_dealer_hedging,
)


def _toy_matrix(spot: float = 200.0) -> dict:
    contracts = []
    for strike, side, oi, gamma, delta in (
        (190, "put", 800, 0.015, -0.4),
        (195, "put", 600, 0.018, -0.35),
        (200, "call", 1000, 0.02, 0.5),
        (205, "call", 700, 0.017, 0.4),
        (210, "call", 500, 0.012, 0.3),
    ):
        contracts.append(
            {
                "strike_price": strike,
                "put_call": side,
                "open_interest": oi,
                "gamma": gamma,
                "delta": delta,
                "volatility": 0.3,
                "days_to_expiration": 14,
                "total_volume": oi // 2,
            }
        )
    return {
        "symbol": "AAPL",
        "underlying_price": spot,
        "as_of_date": "2026-07-31",
        "contracts": contracts,
    }


def test_hedge_curve_from_toy_matrix():
    curve = compute_hedge_demand_curve(_toy_matrix(), n_points=9)
    assert len(curve) == 9
    assert curve[0].spot < 200
    assert curve[-1].spot > 200
    assert any(abs(p.delta_shares) > 0 for p in curve)


def test_simulate_dealer_hedging_outputs():
    sim = simulate_dealer_hedging(_toy_matrix(), n_points=11, expected_move_pct=0.02)
    d = sim.to_dict()
    for key in DEALER_ESTIMATES + DEALER_OUTPUTS:
        assert key in d, key

    assert d["estimate_labels"]["gamma_ramp"] == "Gamma Ramp"
    assert d["output_labels"]["dealer_flow_map"] == "Dealer Flow Map"
    assert len(sim.dealer_flow_map) == 11
    assert len(sim.dealer_demand_curve) == 11
    assert len(sim.dynamic_gamma) == 11
    assert sim.dealer_share_purchases >= 0
    assert sim.dealer_share_sales >= 0
    assert 0 <= sim.dealer_liquidity <= 1
    assert 0 <= sim.dealer_exhaustion <= 1
    assert "shares" in sim.expected_hedging_volume
    assert sim.expected_hedging_volume["shares"] >= 0
    assert "call_wall" in sim.strike_migration["spot"]
    assert set(sim.dealer_flow_map[0]) >= {
        "spot",
        "dealer_share_purchases",
        "dealer_share_sales",
        "net_flow",
        "gamma_exposure",
    }


def test_position_flip_structure():
    sim = simulate_dealer_hedging(_toy_matrix(), n_points=17)
    flip = sim.dealer_position_flip
    assert "gamma_flip_spot" in flip
    assert "hedge_flip_spot" in flip
    assert "distance_pct" in flip


def test_locked_constants():
    assert DEALER_ESTIMATE_LABELS["dealer_share_purchases"] == "Dealer Share Purchases"
    assert DEALER_ESTIMATE_LABELS["dealer_position_flip"] == "Dealer Position Flip"
    assert DEALER_OUTPUTS == (
        "dealer_flow_map",
        "dealer_demand_curve",
        "expected_hedging_volume",
    )
    assert DEALER_OUTPUT_LABELS["expected_hedging_volume"] == "Expected Hedging Volume"


def test_dealer_api_meta():
    from fastapi.testclient import TestClient

    from services.dealer_hedging.app import app

    client = TestClient(app)
    meta = client.get("/v1/meta")
    assert meta.status_code == 200
    body = meta.json()
    assert body["estimates"] == list(DEALER_ESTIMATES)
    assert body["outputs"] == list(DEALER_OUTPUTS)
    assert body["estimate_labels"]["delta_hedging"] == "Delta Hedging"
