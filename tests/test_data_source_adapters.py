"""Phase 3 — Data Sources adapter catalog (Alpaca-primary options; no Schwab)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from gamma_squeeze.data_sources.calendar.opex import third_friday
from gamma_squeeze.data_sources.calendar.political_cycle import first_tuesday_after_first_monday
from gamma_squeeze.data_sources.catalog import LOCKED_ADAPTER_IDS, locked_catalog
from gamma_squeeze.data_sources.registry import catalog, get_adapter, list_adapters, load_all_adapters

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_LOCKED = {
    "options.alpaca",
    "options.ibkr",
    "options.opra",
    "options.polygon",
    "options.cboe",
    "market.ohlcv",
    "market.tick",
    "market.vwap",
    "market.order_book",
    "market.volume_profile",
    "macro.fred",
    "macro.treasury",
    "macro.federal_reserve",
    "macro.economic_calendar",
    "structure.vix",
    "structure.vvix",
    "structure.move",
    "structure.dxy",
    "structure.usdjpy",
    "structure.treasury_yields",
    "corporate.earnings",
    "corporate.dividends",
    "corporate.splits",
    "calendar.opex",
    "calendar.monthly_expiration",
    "calendar.quarterly_expiration",
    "calendar.presidential_cycle",
    "calendar.midterm_cycle",
}


def test_locked_catalog_matches_spec():
    assert set(LOCKED_ADAPTER_IDS) == EXPECTED_LOCKED
    assert "options.schwab" not in LOCKED_ADAPTER_IDS
    doc = locked_catalog()
    assert doc["phase"] == 3


def test_all_locked_adapters_register():
    load_all_adapters()
    names = set(list_adapters())
    assert EXPECTED_LOCKED.issubset(names)
    assert "options.schwab" not in names
    cats = catalog()
    assert "options.alpaca" in cats["options"]
    assert "structure.vix" in cats["structure"]


def test_alpaca_adapter_documents_kv_backup():
    load_all_adapters()
    h = get_adapter("options.alpaca").health()
    assert h["backup_kv_namespace"] == "alpaca-options-matrix-backup"
    assert "alpaca_api" in h["fetch_order"]
    assert "alpaca-options-matrix-backup" in h["fetch_order"]


def test_adapter_health_methods():
    load_all_adapters()
    for name in LOCKED_ADAPTER_IDS:
        h = get_adapter(name).health()
        assert "adapter" in h or "configured" in h


def test_data_sources_doc_alpaca_primary():
    text = (ROOT / "docs" / "DATA_SOURCES.md").read_text(encoding="utf-8")
    assert "alpaca-options-matrix-backup" in text
    assert "Schwab" not in text or "not used" in text.lower()
    assert "Alpaca" in text


def test_opex_third_friday():
    assert third_friday(2026, 1).isoformat() == "2026-01-16"


def test_political_election_day():
    assert first_tuesday_after_first_monday(2024).isoformat() == "2024-11-05"


def test_calendar_adapters_fetch():
    load_all_adapters()
    monthly = get_adapter("calendar.monthly_expiration").fetch(year=2026)
    assert monthly["data"]["monthly"]
    quarterly = get_adapter("calendar.quarterly_expiration").fetch(year=2026)
    assert len(quarterly["data"]["quarterly"]) == 4


def test_structure_symbol_adapters():
    load_all_adapters()
    out = get_adapter("structure.vix").fetch()
    assert out["success"] is True


def test_ibkr_degrades_without_keys():
    load_all_adapters()
    ibkr = get_adapter("options.ibkr").fetch_matrix("AAPL")
    assert ibkr["configured"] is False or ibkr.get("meta", {}).get("status") == "stub"


def test_data_collection_adapters_api_phase():
    from services.data_collection.app import app

    client = TestClient(app)
    resp = client.get("/v1/adapters")
    assert resp.status_code == 200
    body = resp.json()
    assert body["phase"] == 3
    assert body["locked_ok"] is True
    assert "options.schwab" not in body["adapters"]
