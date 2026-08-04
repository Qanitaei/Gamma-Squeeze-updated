import json
from pathlib import Path

import pytest

from gamma_squeeze.features.macro_features import (
    MACRO_FEATURE_COLUMNS,
    MACRO_FEATURES_FIELDS,
    compute_macro_features,
)
from gamma_squeeze.ingest import macro_client


def _write_series(root: Path, series_id: str, rows: list[tuple[str, float]]) -> None:
    payload = [{"DATE": d, "VALUE": str(v), "": ""} for d, v in rows]
    (root / f"{series_id}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def fred_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fred"
    root.mkdir()
    _write_series(
        root,
        "FEDFUNDS",
        [("2026-05-01", 4.33), ("2026-06-01", 4.33), ("2026-07-01", 4.33)],
    )
    _write_series(
        root,
        "CPIAUCSL",
        [(f"2025-{m:02d}-01", 300 + m) for m in range(1, 13)]
        + [("2026-01-01", 313), ("2026-02-01", 314), ("2026-07-01", 320)],
    )
    _write_series(root, "CPILFESL", [("2026-06-01", 310), ("2026-07-01", 311)])
    _write_series(
        root,
        "PPIACO",
        [(f"2025-{m:02d}-01", 250 + m) for m in range(1, 13)] + [("2026-07-01", 265)],
    )
    _write_series(root, "PAYEMS", [("2026-05-01", 158000), ("2026-06-01", 158200), ("2026-07-01", 158500)])
    _write_series(root, "GACDISA066MSFRBNY", [("2026-06-01", 10.0), ("2026-07-01", 15.6)])
    _write_series(root, "A191RL1Q225SBEA", [("2025-10-01", 2.1), ("2026-01-01", 2.4), ("2026-04-01", 2.0)])
    _write_series(root, "UMCSENT", [("2026-06-01", 60.0), ("2026-07-01", 61.7)])
    _write_series(root, "DGS2", [("2026-07-30", 3.8), ("2026-07-31", 3.81)])
    _write_series(root, "DGS10", [("2026-07-30", 4.2), ("2026-07-31", 4.22)])
    _write_series(root, "DGS30", [("2026-07-30", 4.5), ("2026-07-31", 4.51)])
    _write_series(root, "T10Y2Y", [("2026-07-30", 0.40), ("2026-07-31", 0.41)])
    _write_series(root, "T10Y3M", [("2026-07-31", 0.92)])
    _write_series(root, "DTWEXBGS", [("2026-07-24", 120.7)])
    _write_series(root, "DEXJPUS", [("2026-07-24", 163.71)])
    _write_series(root, "VIXCLS", [("2026-07-30", 17.09)])
    _write_series(root, "BAA10Y", [("2026-07-30", 1.64)])
    _write_series(root, "BAMLC0A0CM", [("2026-07-30", 0.80)])
    _write_series(root, "BAMLH0A0HYM2", [("2026-07-30", 2.84)])
    _write_series(root, "MOVE", [("2026-07-30", 95.2)])
    _write_series(
        root,
        "WEI",
        [(f"2025-{(i % 12) + 1:02d}-0{(i % 2) + 1}", 1.0 + 0.05 * i) for i in range(60)],
    )
    _write_series(root, "STLFSI4", [("2026-07-24", -0.8)])

    monkeypatch.setattr(macro_client, "local_fred_roots", lambda: [root])
    monkeypatch.setattr(macro_client, "fred_worker_url", lambda: "http://127.0.0.1:9")
    macro_client.clear_fred_series_cache()
    return root


def test_macro_features_locked_catalog(fred_root: Path):
    feats = compute_macro_features(as_of="2026-07-31", include_external_vol=False)
    catalog = feats.catalog_dict()
    for key in MACRO_FEATURES_FIELDS:
        assert key in catalog, key
        # MOVE present from local KV; VVIX needs external and may be None when disabled
        if key == "vvix":
            continue
        assert catalog[key] is not None, key

    assert feats.fed_funds == pytest.approx(4.33)
    assert feats.yield_spread == pytest.approx(0.41)
    assert feats.treasury_curve["10y"] == pytest.approx(4.22)
    assert feats.vix == pytest.approx(17.09)
    assert feats.high_yield_spread == pytest.approx(2.84)
    assert feats.move == pytest.approx(95.2)
    assert feats.cpi_yoy is not None
    assert feats.payrolls_mom is not None

    flat = feats.flat_features()
    for col in MACRO_FEATURE_COLUMNS:
        assert col in flat
    assert "meta" not in flat
    assert "treasury_curve" not in flat


def test_macro_point_in_time(fred_root: Path):
    early = compute_macro_features(as_of="2026-06-15", include_external_vol=False)
    late = compute_macro_features(as_of="2026-07-31", include_external_vol=False)
    assert early.pmi == pytest.approx(10.0)
    assert late.pmi == pytest.approx(15.6)
    assert early.yield_spread in (None, 0.40) or early.yield_spread <= late.yield_spread


def test_macro_feature_block_service(fred_root: Path):
    from services.feature_engineering.service import compute_macro_feature_block

    out = compute_macro_feature_block(as_of="2026-07-31", include_external_vol=False)
    assert out["success"] is True
    assert out["feature_names"] == list(MACRO_FEATURES_FIELDS)
    for key in MACRO_FEATURES_FIELDS:
        if key == "vvix":
            continue
        assert out["macro_features"][key] is not None
