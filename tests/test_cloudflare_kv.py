import json
import math

import pytest

from gamma_squeeze.cloudflare_kv import (
    clean_api_token,
    dumps_kv_json,
    gamma_squeeze_namespace_id,
    sanitize_for_json,
)


@pytest.mark.unit
def test_clean_api_token_strips_yaml_list_corruption():
    # Use a non-provider-shaped fixture so secret scanners ignore it.
    raw = "\t- unit_test_token_value_000000000000000000000000"
    cleaned = clean_api_token(raw)
    assert cleaned == "unit_test_token_value_000000000000000000000000"
    assert "\t" not in cleaned
    assert not cleaned.startswith("-")


@pytest.mark.unit
def test_clean_api_token_empty():
    assert clean_api_token("") == ""
    assert clean_api_token("   ") == ""


@pytest.mark.unit
def test_gamma_squeeze_namespace_id_default():
    ns = gamma_squeeze_namespace_id()
    assert ns == "f949a0301f604312a9c8959f6f2a3918"


@pytest.mark.unit
def test_dumps_kv_json_nulls_nonfinite_floats():
    payload = {"anchor_value": float("nan"), "hi": float("inf"), "ok": 1.25}
    cleaned = sanitize_for_json(payload)
    assert cleaned["anchor_value"] is None
    assert cleaned["hi"] is None
    assert cleaned["ok"] == 1.25
    text = dumps_kv_json(payload)
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text)["anchor_value"] is None
    assert not any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in cleaned.values())


@pytest.mark.unit
def test_sync_same_store_does_not_raise(tmp_path):
    from gamma_squeeze.scan.confirmed_squeeze import sync_matrices_to_ssd

    tickers = tmp_path / "tickers" / "AAPL"
    tickers.mkdir(parents=True)
    (tickers / "2026-08-07.json").write_text('{"symbol":"AAPL","contracts":[]}')
    info = sync_matrices_to_ssd(
        ["AAPL"],
        source_root=tmp_path / "tickers",
        dest_root=tmp_path,
        latest_only=True,
        fetch_missing_from_kv=False,
    )
    assert info.get("same_store") is True
    assert info["copied_files"] == 0
