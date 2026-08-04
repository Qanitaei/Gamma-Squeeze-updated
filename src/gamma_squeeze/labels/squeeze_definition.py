"""Explicit, backtestable gamma-squeeze label definition.

A candidate day is "fragile" when dealer gamma positioning is stressed.
A forward squeeze outcome at horizon h is a large upside move with optional
volume confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass

# Candidate (setup) thresholds
MIN_POSITIONING_STRESS = 0.35
FRAGILE_GEX_REQUIRED = True  # prefer net GEX <= 0 via flag_fragile_gamma

# Outcome thresholds (forward returns)
UPSIDE_MOVE_PCT = {
    1: 0.025,
    2: 0.035,
    3: 0.045,
    4: 0.05,
    5: 0.055,
    6: 0.06,
    7: 0.065,
    8: 0.07,
    9: 0.075,
    10: 0.08,
}


@dataclass(frozen=True)
class SqueezeLabelSpec:
    min_positioning_stress: float = MIN_POSITIONING_STRESS
    require_fragile_gex: bool = FRAGILE_GEX_REQUIRED
    upside_move_pct: dict[int, float] | None = None

    def threshold_for(self, horizon: int) -> float:
        table = self.upside_move_pct or UPSIDE_MOVE_PCT
        return float(table.get(horizon, 0.05 + 0.005 * horizon))


DEFAULT_SPEC = SqueezeLabelSpec()


def is_candidate_setup(row: dict) -> bool:
    stress = float(row.get("positioning_stress") or 0.0)
    if stress < DEFAULT_SPEC.min_positioning_stress:
        return False
    if DEFAULT_SPEC.require_fragile_gex and float(row.get("flag_fragile_gamma") or 0.0) < 0.5:
        # still allow near-flip or call crowding
        if float(row.get("flag_near_flip") or 0.0) < 0.5 and float(row.get("flag_call_crowding") or 0.0) < 0.5:
            return False
    return True


def is_squeeze_outcome(fwd_return: float | None, horizon: int, spec: SqueezeLabelSpec = DEFAULT_SPEC) -> bool:
    if fwd_return is None:
        return False
    return float(fwd_return) >= spec.threshold_for(horizon)
