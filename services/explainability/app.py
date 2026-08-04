"""Explainability Engine API — port 8013."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.common.app_factory import create_service_app
from services.common.schemas import ServiceEnvelope
from services.explainability.service import explain_meta, run_explain

SERVICE = "explainability"
app = create_service_app(
    name=SERVICE,
    description=(
        "Explainability engine — every prediction explains WHY: SHAP, Attention Maps, "
        "Feature Importance, Counterfactual Analysis, Partial Dependence, Decision Trace, "
        "Model Confidence"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class ExplainRequest(BaseModel):
    symbol: str
    as_of: str | None = None
    include_forecast: bool = Field(True, description="Run prediction then explain it")
    use_composite: bool = Field(True, description="Prefer composite squeeze prediction")
    use_kernel_shap: bool = Field(False, description="Opt-in slow KernelExplainer")


@router.get("/meta")
def meta() -> dict:
    return explain_meta()


@router.post("/explain", response_model=ServiceEnvelope)
def explain(req: ExplainRequest) -> ServiceEnvelope:
    data = run_explain(
        req.symbol,
        as_of=req.as_of,
        include_forecast=req.include_forecast,
        use_composite=req.use_composite,
        use_kernel_shap=req.use_kernel_shap,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=bool(data.get("error")) or not bool(data.get("why")),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/explain/{symbol}", response_model=ServiceEnvelope)
def explain_get(
    symbol: str,
    as_of: str | None = None,
    include_forecast: bool = True,
    use_composite: bool = True,
    use_kernel_shap: bool = False,
) -> ServiceEnvelope:
    return explain(
        ExplainRequest(
            symbol=symbol,
            as_of=as_of,
            include_forecast=include_forecast,
            use_composite=use_composite,
            use_kernel_shap=use_kernel_shap,
        )
    )


app.include_router(router)
