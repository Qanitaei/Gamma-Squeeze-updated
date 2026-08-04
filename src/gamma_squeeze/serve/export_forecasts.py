"""Persist SqueezeForecast JSON to SSD / local exports."""

from __future__ import annotations

import json
from pathlib import Path

from gamma_squeeze.config import resolve_forecasts_root
from gamma_squeeze.serve.schemas import SqueezeForecast, validate_forecast_dict


def forecast_paths(symbol: str, as_of: str, *, root: Path | None = None) -> tuple[Path, Path]:
    base = root or resolve_forecasts_root()
    dated = base / "by-date" / as_of / f"{symbol.upper()}.json"
    latest = base / "by-ticker" / symbol.upper() / "latest.json"
    return dated, latest


def export_forecast(forecast: SqueezeForecast, *, root: Path | None = None) -> Path:
    payload = forecast.to_dict()
    errors = validate_forecast_dict(payload)
    if errors:
        raise ValueError("Invalid forecast payload: " + "; ".join(errors[:8]))

    dated, latest = forecast_paths(forecast.symbol, forecast.as_of, root=root)
    for path in (dated, latest):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
    return dated


def load_latest_forecast(symbol: str, *, root: Path | None = None) -> dict | None:
    _, latest = forecast_paths(symbol, "1970-01-01", root=root)
    # forecast_paths needs as_of only for dated; latest path ignores as_of content except parent
    latest = (root or resolve_forecasts_root()) / "by-ticker" / symbol.upper() / "latest.json"
    if not latest.is_file():
        return None
    with latest.open() as f:
        return json.load(f)
