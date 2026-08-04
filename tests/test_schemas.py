from gamma_squeeze.config import HORIZONS
from gamma_squeeze.serve.schemas import (
    build_empty_forecast,
    validate_forecast_dict,
)


def test_empty_forecast_valid():
    fc = build_empty_forecast("AAPL", "2026-07-31")
    errors = validate_forecast_dict(fc.to_dict())
    assert errors == []


def test_horizons_cover_1_to_10():
    fc = build_empty_forecast("MSFT", "2026-07-31")
    hs = [h.horizon_days for h in fc.horizons]
    assert hs == list(range(1, 11))
    assert tuple(hs) == HORIZONS


def test_invalid_probability_caught():
    fc = build_empty_forecast("NVDA", "2026-07-31")
    d = fc.to_dict()
    d["horizons"][0]["squeeze_probability"] = 1.5
    errors = validate_forecast_dict(d)
    assert any("squeeze_probability" in e for e in errors)
