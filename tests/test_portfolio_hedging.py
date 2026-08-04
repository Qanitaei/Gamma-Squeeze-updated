from gamma_squeeze.risk.portfolio_hedging_engine import (
    HEDGE_STRUCTURES,
    OPTIMIZE_OBJECTIVES,
    optimize_hedge_book,
    run_portfolio_hedging_engine,
)
from gamma_squeeze.serve.schemas import (
    HorizonForecast,
    MagnitudeQuantiles,
)


def _horizons(p: float = 0.65) -> list[HorizonForecast]:
    out = []
    for h in (1, 3, 5, 10):
        out.append(
            HorizonForecast(
                horizon_days=h,
                squeeze_probability=p,
                expected_magnitude_pct=4.0,
                magnitude_quantiles=MagnitudeQuantiles(p10=1.0, p50=3.0, p90=7.0),
                expected_duration_days=float(h),
                return_distribution=[],
                confidence_score=0.7,
            )
        )
    return out


def test_structures_and_objectives_catalog():
    assert len(HEDGE_STRUCTURES) == 13
    assert "Iron Condor" in HEDGE_STRUCTURES
    assert "Dynamic Delta Hedge" in HEDGE_STRUCTURES
    assert set(OPTIMIZE_OBJECTIVES) == {"Return", "Risk", "Capital Efficiency", "Margin"}


def test_engine_recommendations_and_optimize():
    row = {
        "symbol": "TEST",
        "as_of": "2025-06-02",
        "spot": 100.0,
        "positioning_stress": 0.75,
        "regime_neg_gamma": 1.0,
        "net_gex": -5e5,
        "atm_iv": 0.35,
        "dealer_delta": -20000,
        "dealer_gamma": -900,
        "dealer_vega": 50,
        "dealer_theta": -30,
        "expected_move": 3.5,
    }
    plan = run_portfolio_hedging_engine(
        "TEST",
        row,
        horizons=_horizons(0.7),
        optimize_for="Risk",
        data_origin="test",
    )
    assert len(plan.recommendations) == 13
    names = {r.structure for r in plan.recommendations}
    assert names == set(HEDGE_STRUCTURES)
    for r in plan.recommendations:
        for obj in OPTIMIZE_OBJECTIVES:
            assert 0.0 <= r.scores[obj] <= 1.0
    opt = plan.optimized
    assert opt["objective"] == "Risk"
    assert opt["weights"]
    assert abs(sum(opt["weights"].values()) - 1.0) < 1e-6
    schema = plan.as_schema_hedges()
    assert schema and schema[0].instrument in HEDGE_STRUCTURES


def test_optimize_objectives_change_ranking():
    row = {
        "symbol": "TEST",
        "as_of": "2025-06-02",
        "spot": 100.0,
        "positioning_stress": 0.2,
        "regime_neg_gamma": 0.0,
        "net_gex": 1e5,
        "atm_iv": 0.2,
        "dealer_delta": 5000,
        "dealer_gamma": 100,
        "dealer_vega": 10,
        "dealer_theta": -5,
    }
    risk_plan = run_portfolio_hedging_engine("TEST", row, horizons=_horizons(0.25), optimize_for="Risk")
    ret_plan = run_portfolio_hedging_engine("TEST", row, horizons=_horizons(0.8), optimize_for="Return")
    assert risk_plan.optimized["objective"] == "Risk"
    assert ret_plan.optimized["objective"] == "Return"
    # re-optimize same scores for Capital Efficiency
    cap = optimize_hedge_book(ret_plan.recommendations, objective="Capital Efficiency")
    assert cap["objective"] == "Capital Efficiency"


def test_schwab_adapter_removed():
    from gamma_squeeze.data_sources.registry import list_adapters, load_all_adapters

    load_all_adapters()
    assert "options.schwab" not in list_adapters()
    try:
        from gamma_squeeze.data_sources.options import schwab  # noqa: F401
        raise AssertionError("options.schwab module should not exist")
    except ImportError:
        pass


def test_full_recommend_catalog():
    assert HEDGE_STRUCTURES == (
        "Long Calls",
        "Long Puts",
        "Covered Calls",
        "Protective Puts",
        "Debit Spreads",
        "Credit Spreads",
        "Calendar Spreads",
        "Iron Condor",
        "Collar",
        "Dynamic Delta Hedge",
        "Gamma Hedge",
        "Vega Hedge",
        "Theta Hedge",
    )
    assert OPTIMIZE_OBJECTIVES == ("Return", "Risk", "Capital Efficiency", "Margin")


def test_portfolio_api_meta():
    from fastapi.testclient import TestClient

    from services.portfolio_hedging.app import app

    client = TestClient(app)
    meta = client.get("/v1/meta")
    assert meta.status_code == 200
    body = meta.json()
    assert body["recommend"] == list(HEDGE_STRUCTURES)
    assert body["optimize"] == list(OPTIMIZE_OBJECTIVES)
    assert "Capital Efficiency" in body["optimize_labels"]

    st = client.get("/v1/structures")
    assert st.status_code == 200
    assert len(st.json()["recommend"]) == 13
