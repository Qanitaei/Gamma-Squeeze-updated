import json
from pathlib import Path

import pytest

from gamma_squeeze.config import FALLBACK_ROOT, resolve_matrix_root
from gamma_squeeze.scan.confirmed_squeeze import (
    CONFIRMED_THRESHOLD,
    DEFAULT_TOP100_CACHE,
    load_top_market_cap,
    sync_matrices_to_ssd,
)


@pytest.mark.unit
def test_load_top100_cache():
    rows = load_top_market_cap(top=100, refresh=False)
    assert len(rows) == 100
    assert rows[0]["symbol"]
    assert float(rows[0].get("market_cap") or 0) >= float(rows[-1].get("market_cap") or 0)


@pytest.mark.unit
def test_sync_matrices_smoke(tmp_path):
    candidates = [
        resolve_matrix_root() / "tickers",
        FALLBACK_ROOT / "tickers",
        Path("/Volumes/PortableSSD/Gamma Squeeze Matrix/tickers"),
    ]
    src = next((p for p in candidates if p.is_dir() and any(p.glob("*/*.json"))), None)
    if src is None:
        pytest.skip("alpaca matrix store missing")
    # sync_matrices_to_ssd expects SYMBOL/*.json (no tickers/ prefix) OR we pass tickers parent
    # Prefer a store that has SYMBOL subdirs with json files.
    info = sync_matrices_to_ssd(
        ["AAPL", "MSFT"],
        source_root=src,
        dest_root=tmp_path,
        latest_only=True,
        fetch_missing_from_kv=False,
    )
    # When source is tickers/, sync looks for src/AAPL — which works if src is the tickers dir.
    assert info["copied_files"] >= 1 or (tmp_path / "tickers" / "AAPL").is_dir()
    if info["copied_files"] >= 1:
        assert (tmp_path / "tickers" / "AAPL").is_dir()


@pytest.mark.unit
def test_default_top100_cache_in_platform():
    assert "schwab-options-export" not in str(DEFAULT_TOP100_CACHE)
    assert DEFAULT_TOP100_CACHE.name == "top100_nasdaq_by_market_cap.json"


@pytest.mark.unit
def test_confirmed_threshold_constant():
    assert 0.5 <= CONFIRMED_THRESHOLD <= 0.8


@pytest.mark.integration
def test_latest_scan_export_shape():
    latest = Path(
        "/Users/ruslantkach/Desktop/gamma-squeeze-platform/data/exports/"
        "Gamma Squeeze Matrix/scans/confirmed_gamma_squeeze/latest.json"
    )
    if not latest.is_file():
        pytest.skip("scan not run yet")
    payload = json.loads(latest.read_text())
    assert payload["n_scanned"] >= 1
    assert "gex" in payload["variables"]
    assert "candlestick_patterns" in payload["variables"]
    if payload.get("results"):
        row = payload["results"][0]
        assert "greeks" in row and "dex" in row and "patterns" in row
