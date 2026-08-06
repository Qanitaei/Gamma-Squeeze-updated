"""Forecast JSON schema and validators for gamma-squeeze outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from gamma_squeeze.config import HORIZONS

SCHEMA_VERSION = "1.0.0"


@dataclass
class MagnitudeQuantiles:
    p10: float
    p50: float
    p90: float


@dataclass
class DistributionBin:
    lo_pct: float
    hi_pct: float
    probability: float


@dataclass
class HedgeDemandPoint:
    spot: float
    delta_shares: float
    delta_dollars: float
    gamma_exposure: float


@dataclass
class TradeRecommendation:
    structure: str
    direction: str
    size_hint: str
    risk_score: float
    rationale: str
    horizon_days: int


@dataclass
class PortfolioHedgeRecommendation:
    instrument: str
    action: str
    size_hint: str
    hedge_ratio: float
    rationale: str


@dataclass
class Explanation:
    feature: str
    contribution: float
    direction: str
    note: str


@dataclass
class HorizonForecast:
    horizon_days: int
    squeeze_probability: float
    expected_magnitude_pct: float
    magnitude_quantiles: MagnitudeQuantiles
    expected_duration_days: float
    return_distribution: list[DistributionBin]
    confidence_score: float


@dataclass
class SqueezeForecast:
    schema_version: str
    symbol: str
    as_of: str  # market / feature date YYYY-MM-DD
    horizons: list[HorizonForecast]
    dealer_hedging_demand: list[HedgeDemandPoint]
    trade_recommendations: list[TradeRecommendation]
    portfolio_hedge_recommendations: list[PortfolioHedgeRecommendation]
    explanations: list[Explanation]
    model_versions: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    extracted_at: str = ""  # UTC ISO-8601 with time (day of extraction)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_keys(obj: dict[str, Any], keys: list[str], ctx: str) -> list[str]:
    return [f"{ctx}: missing '{k}'" for k in keys if k not in obj]


def validate_forecast_dict(payload: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty if valid)."""
    errors: list[str] = []
    errors.extend(
        _require_keys(
            payload,
            [
                "schema_version",
                "symbol",
                "as_of",
                "horizons",
                "dealer_hedging_demand",
                "trade_recommendations",
                "portfolio_hedge_recommendations",
                "explanations",
                "model_versions",
            ],
            "root",
        )
    )
    # extracted_at recommended (ISO datetime); not hard-fail for older payloads
    ext = payload.get("extracted_at")
    if ext is not None and ext != "" and not isinstance(ext, str):
        errors.append("root: extracted_at must be an ISO-8601 string")
    horizons = payload.get("horizons")
    if not isinstance(horizons, list) or not horizons:
        errors.append("root: horizons must be a non-empty list")
        return errors

    seen: set[int] = set()
    for i, h in enumerate(horizons):
        if not isinstance(h, dict):
            errors.append(f"horizons[{i}]: must be object")
            continue
        errors.extend(
            _require_keys(
                h,
                [
                    "horizon_days",
                    "squeeze_probability",
                    "expected_magnitude_pct",
                    "magnitude_quantiles",
                    "expected_duration_days",
                    "return_distribution",
                    "confidence_score",
                ],
                f"horizons[{i}]",
            )
        )
        hd = h.get("horizon_days")
        if hd not in HORIZONS:
            errors.append(f"horizons[{i}]: horizon_days must be in 1..10, got {hd}")
        if hd in seen:
            errors.append(f"horizons[{i}]: duplicate horizon_days {hd}")
        seen.add(hd) if isinstance(hd, int) else None

        p = h.get("squeeze_probability")
        if isinstance(p, (int, float)) and not (0.0 <= float(p) <= 1.0):
            errors.append(f"horizons[{i}]: squeeze_probability out of [0,1]")
        c = h.get("confidence_score")
        if isinstance(c, (int, float)) and not (0.0 <= float(c) <= 1.0):
            errors.append(f"horizons[{i}]: confidence_score out of [0,1]")

        mq = h.get("magnitude_quantiles")
        if isinstance(mq, dict):
            for q in ("p10", "p50", "p90"):
                if q not in mq:
                    errors.append(f"horizons[{i}].magnitude_quantiles: missing {q}")
        else:
            errors.append(f"horizons[{i}]: magnitude_quantiles must be object")

        rd = h.get("return_distribution")
        if not isinstance(rd, list) or not rd:
            errors.append(f"horizons[{i}]: return_distribution must be a non-empty list")

    for key in (
        "dealer_hedging_demand",
        "trade_recommendations",
        "portfolio_hedge_recommendations",
        "explanations",
    ):
        if key in payload and not isinstance(payload.get(key), list):
            errors.append(f"root: {key} must be a list")

    mv = payload.get("model_versions")
    if mv is not None and not isinstance(mv, dict):
        errors.append("root: model_versions must be an object")

    missing_h = set(HORIZONS) - seen
    if missing_h:
        errors.append(f"root: missing horizons {sorted(missing_h)}")

    return errors


def empty_horizon(horizon_days: int) -> HorizonForecast:
    return HorizonForecast(
        horizon_days=horizon_days,
        squeeze_probability=0.0,
        expected_magnitude_pct=0.0,
        magnitude_quantiles=MagnitudeQuantiles(p10=0.0, p50=0.0, p90=0.0),
        expected_duration_days=float(horizon_days),
        return_distribution=[
            DistributionBin(lo_pct=-5.0, hi_pct=0.0, probability=0.5),
            DistributionBin(lo_pct=0.0, hi_pct=5.0, probability=0.5),
        ],
        confidence_score=0.0,
    )


def build_empty_forecast(symbol: str, as_of: str) -> SqueezeForecast:
    return SqueezeForecast(
        schema_version=SCHEMA_VERSION,
        symbol=symbol.upper(),
        as_of=as_of,
        horizons=[empty_horizon(h) for h in HORIZONS],
        dealer_hedging_demand=[],
        trade_recommendations=[],
        portfolio_hedge_recommendations=[],
        explanations=[],
        model_versions={"platform": "0.1.0"},
        meta={},
    )
