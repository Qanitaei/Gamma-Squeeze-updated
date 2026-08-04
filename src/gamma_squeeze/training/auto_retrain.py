"""Automatic retraining policy driven by drift + schedule."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from gamma_squeeze.training.drift import DriftReport


@dataclass
class RetrainDecision:
    should_retrain: bool
    reasons: list[str] = field(default_factory=list)
    drift: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    decided_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AutoRetrainPolicy:
    on_feature_drift: bool = True
    on_prediction_drift: bool = True
    on_concept_drift: bool = True
    min_reasons: int = 1
    force: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_retrain(
    drift: DriftReport | dict[str, Any],
    *,
    policy: AutoRetrainPolicy | None = None,
) -> RetrainDecision:
    pol = policy or AutoRetrainPolicy()
    d = drift.to_dict() if isinstance(drift, DriftReport) else dict(drift)
    reasons: list[str] = []
    if pol.force:
        reasons.append("forced")
    feat_trig = bool((d.get("feature_drift") or {}).get("triggered"))
    pred_trig = bool((d.get("prediction_drift") or {}).get("triggered"))
    concept_trig = bool((d.get("concept_drift") or {}).get("triggered"))
    if pol.on_feature_drift and feat_trig:
        reasons.append("feature_drift")
    if pol.on_prediction_drift and pred_trig:
        reasons.append("prediction_drift")
    if pol.on_concept_drift and concept_trig:
        reasons.append("concept_drift")
    # dedupe preserve order
    seen = set()
    uniq = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    should = len([r for r in uniq if r != "forced"]) >= pol.min_reasons or pol.force
    if pol.force and "forced" not in uniq:
        uniq.insert(0, "forced")
    return RetrainDecision(
        should_retrain=should,
        reasons=uniq,
        drift=d,
        policy=pol.to_dict(),
        decided_at=datetime.now(timezone.utc).isoformat(),
    )
