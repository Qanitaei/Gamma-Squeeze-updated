"""Feature Engineering API — port 8002."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter
from pydantic import BaseModel

from services.common.app_factory import create_service_app
from services.common.schemas import ServiceEnvelope
from services.feature_engineering.service import (
    build_or_load_features,
    compute_dealer_features,
    compute_macro_feature_block,
    compute_options_metrics_features,
    compute_technical_features,
)

SERVICE = "feature_engineering"
app = create_service_app(
    name=SERVICE,
    description=(
        "Feature Engineering: Dealer Positioning from Alpaca API / "
        "alpaca-options-matrix-backup KV; options metrics, technicals, macro"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class FeatureRequest(BaseModel):
    symbol: str
    rebuild: bool = False
    include_skew: bool = False
    include_macro: bool = True
    include_dealer: bool = True
    include_options_metrics: bool = True
    include_technicals: bool = True


class DealerRequest(BaseModel):
    symbol: str
    as_of: str | None = None
    include_gamma_by_strike: bool = True


class OptionsMetricsRequest(BaseModel):
    symbol: str
    as_of: str | None = None
    include_curves: bool = True


class TechnicalRequest(BaseModel):
    symbol: str
    as_of: str | None = None
    anchor_date: str | None = None
    include_volume_profile: bool = True


class MacroRequest(BaseModel):
    as_of: str | None = None
    include_external_vol: bool = True


@router.post("/features", response_model=ServiceEnvelope)
def features(req: FeatureRequest) -> ServiceEnvelope:
    data = build_or_load_features(
        req.symbol,
        rebuild=req.rebuild,
        include_skew=req.include_skew,
        include_macro=req.include_macro,
        include_dealer=req.include_dealer,
        include_options_metrics=req.include_options_metrics,
        include_technicals=req.include_technicals,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=data.get("n_rows", 0) == 0,
        data=data,
    )


@router.get("/features/{symbol}", response_model=ServiceEnvelope)
def features_get(
    symbol: str,
    rebuild: bool = False,
    include_dealer: bool = True,
    include_options_metrics: bool = True,
    include_technicals: bool = True,
) -> ServiceEnvelope:
    return features(
        FeatureRequest(
            symbol=symbol,
            rebuild=rebuild,
            include_dealer=include_dealer,
            include_options_metrics=include_options_metrics,
            include_technicals=include_technicals,
        )
    )


@router.post("/dealer-positioning", response_model=ServiceEnvelope)
def dealer_positioning(req: DealerRequest) -> ServiceEnvelope:
    data = compute_dealer_features(
        req.symbol,
        as_of=req.as_of,
        include_gamma_by_strike=req.include_gamma_by_strike,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=not data.get("success", False),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/dealer-positioning/{symbol}", response_model=ServiceEnvelope)
def dealer_positioning_get(
    symbol: str,
    as_of: str | None = None,
    include_gamma_by_strike: bool = True,
) -> ServiceEnvelope:
    return dealer_positioning(
        DealerRequest(
            symbol=symbol,
            as_of=as_of,
            include_gamma_by_strike=include_gamma_by_strike,
        )
    )


@router.post("/options-metrics", response_model=ServiceEnvelope)
def options_metrics(req: OptionsMetricsRequest) -> ServiceEnvelope:
    data = compute_options_metrics_features(
        req.symbol,
        as_of=req.as_of,
        include_curves=req.include_curves,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=not data.get("success", False),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/options-metrics/{symbol}", response_model=ServiceEnvelope)
def options_metrics_get(
    symbol: str,
    as_of: str | None = None,
    include_curves: bool = True,
) -> ServiceEnvelope:
    return options_metrics(
        OptionsMetricsRequest(symbol=symbol, as_of=as_of, include_curves=include_curves)
    )


@router.post("/technical-indicators", response_model=ServiceEnvelope)
def technical_indicators(req: TechnicalRequest) -> ServiceEnvelope:
    data = compute_technical_features(
        req.symbol,
        as_of=req.as_of,
        anchor_date=req.anchor_date,
        include_volume_profile=req.include_volume_profile,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=not data.get("success", False),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/technical-indicators/{symbol}", response_model=ServiceEnvelope)
def technical_indicators_get(
    symbol: str,
    as_of: str | None = None,
    anchor_date: str | None = None,
    include_volume_profile: bool = True,
) -> ServiceEnvelope:
    return technical_indicators(
        TechnicalRequest(
            symbol=symbol,
            as_of=as_of,
            anchor_date=anchor_date,
            include_volume_profile=include_volume_profile,
        )
    )


@router.post("/macro-features", response_model=ServiceEnvelope)
def macro_features(req: MacroRequest) -> ServiceEnvelope:
    data = compute_macro_feature_block(
        as_of=req.as_of,
        include_external_vol=req.include_external_vol,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=None,
        as_of=data.get("as_of"),
        degraded=not data.get("success", False),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/macro-features", response_model=ServiceEnvelope)
def macro_features_get(
    as_of: str | None = None,
    include_external_vol: bool = True,
) -> ServiceEnvelope:
    return macro_features(MacroRequest(as_of=as_of, include_external_vol=include_external_vol))


app.include_router(router)
