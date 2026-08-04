"""Discrete action catalog for the PPO trading agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    id: int
    name: str
    family: str  # directional | options | hedge | cash
    description: str


ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec(0, "Long", "directional", "Long underlying / squeeze exposure"),
    ActionSpec(1, "Short", "directional", "Short underlying"),
    ActionSpec(2, "Calls", "options", "Long call(s)"),
    ActionSpec(3, "Puts", "options", "Long put(s)"),
    ActionSpec(4, "Debit Spread", "options", "Bull/bear debit vertical"),
    ActionSpec(5, "Credit Spread", "options", "Bull/bear credit vertical"),
    ActionSpec(6, "Iron Condor", "options", "Short vol iron condor"),
    ActionSpec(7, "Calendar", "options", "Calendar / diagonal"),
    ActionSpec(8, "Butterfly", "options", "Long butterfly"),
    ActionSpec(9, "Covered Call", "options", "Long stock + short call"),
    ActionSpec(10, "Cash", "cash", "Flat / raise cash"),
    ActionSpec(11, "Increase Hedge", "hedge", "Increase hedge overlay"),
    ActionSpec(12, "Reduce Hedge", "hedge", "Reduce hedge overlay"),
)

ACTION_NAMES: dict[int, str] = {a.id: a.name for a in ACTIONS}
ACTION_BY_NAME: dict[str, int] = {a.name: a.id for a in ACTIONS}
N_ACTIONS = len(ACTIONS)


def action_meta() -> list[dict[str, str | int]]:
    return [
        {"id": a.id, "name": a.name, "family": a.family, "description": a.description}
        for a in ACTIONS
    ]
