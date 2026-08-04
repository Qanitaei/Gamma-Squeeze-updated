"""PPO Reinforcement Learning Agent API — port 8009."""

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
from services.ppo_agent.service import ppo_meta, run_ppo

SERVICE = "ppo_agent"
app = create_service_app(
    name=SERVICE,
    description=(
        "Stable-Baselines3 PPO agent — state: dealer/regime/forecast/portfolio/risk/"
        "exposure/IV/greeks/macro; actions: Long/Short/Calls/Puts/spreads/condor/"
        "calendar/butterfly/covered/cash/hedge; reward: Sharpe/Sortino/DD/costs/"
        "slippage/tail/variance"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class PPORequest(BaseModel):
    symbol: str
    train: bool = False
    timesteps: int = Field(2048, ge=128, le=200_000)
    horizon: int = Field(5, ge=1, le=10)
    deterministic: bool = True
    use_composite_forecast: bool = False
    # legacy alias
    train_episodes: int | None = Field(None, description="Deprecated; use train+timesteps")


@router.get("/meta")
def meta() -> dict:
    return ppo_meta()


@router.post("/action", response_model=ServiceEnvelope)
def action(req: PPORequest) -> ServiceEnvelope:
    train = req.train or (req.train_episodes is not None and req.train_episodes > 0)
    # Map legacy episodes → short timesteps
    timesteps = req.timesteps
    if req.train_episodes is not None and req.train_episodes > 0 and not req.train:
        timesteps = max(256, int(req.train_episodes) * 256)
        train = True
    data = run_ppo(
        req.symbol,
        train=train,
        timesteps=timesteps,
        horizon=req.horizon,
        deterministic=req.deterministic,
        use_composite_forecast=req.use_composite_forecast,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=bool(data.get("error")),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/action/{symbol}", response_model=ServiceEnvelope)
def action_get(
    symbol: str,
    train: bool = False,
    timesteps: int = 2048,
    horizon: int = 5,
    deterministic: bool = True,
    use_composite_forecast: bool = False,
    train_episodes: int | None = None,
) -> ServiceEnvelope:
    return action(
        PPORequest(
            symbol=symbol,
            train=train,
            timesteps=timesteps,
            horizon=horizon,
            deterministic=deterministic,
            use_composite_forecast=use_composite_forecast,
            train_episodes=train_episodes,
        )
    )


@router.post("/train", response_model=ServiceEnvelope)
def train(req: PPORequest) -> ServiceEnvelope:
    req.train = True
    return action(req)


app.include_router(router)
