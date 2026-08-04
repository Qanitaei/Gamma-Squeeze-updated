from __future__ import annotations

from typing import Any

from gamma_squeeze.viz.institutional_dashboard import (
    build_institutional_dashboard,
    dashboard_meta,
)


def build_dashboard(
    symbol: str,
    *,
    use_live_alpaca: bool = True,
    full: bool = False,
) -> dict[str, Any]:
    """Institutional dashboard panels + heatmaps (Alpaca-forward, JSON only)."""
    data = build_institutional_dashboard(
        symbol,
        use_live_alpaca=use_live_alpaca,
        include_pipeline_extras=full,
    )
    data.pop("html", None)
    return data


def meta() -> dict[str, Any]:
    """Meta via Alpaca API health — not a local curl stub."""
    return dashboard_meta()
