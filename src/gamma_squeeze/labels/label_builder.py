"""Build multi-horizon squeeze labels from features + OHLCV forward returns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from gamma_squeeze.config import HORIZONS, resolve_features_root

# Include 20d for direction model even though squeeze horizons are 1–10
_LABEL_HORIZONS = tuple(sorted(set(HORIZONS) | {20}))
from gamma_squeeze.features.feature_store import load_features
from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv
from gamma_squeeze.labels.squeeze_definition import (
    DEFAULT_SPEC,
    SqueezeLabelSpec,
    is_candidate_setup,
    is_squeeze_outcome,
)


def labels_path(symbol: str, *, root: Path | None = None) -> Path:
    base = root or resolve_features_root()
    return base / "by_ticker" / symbol.upper() / "labels.json"


def save_labels(symbol: str, df: pd.DataFrame, *, root: Path | None = None) -> Path:
    path = labels_path(symbol, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": symbol.upper(),
        "n_rows": int(len(df)),
        "horizons": list(_LABEL_HORIZONS),
        "rows": df.where(pd.notnull(df), None).to_dict(orient="records"),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_labels(symbol: str, *, root: Path | None = None) -> pd.DataFrame:
    path = labels_path(symbol, root=root)
    if not path.is_file():
        return pd.DataFrame()
    with path.open() as f:
        payload = json.load(f)
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    return pd.DataFrame(rows or [])


def build_labels_for_symbol(
    symbol: str,
    *,
    features: pd.DataFrame | None = None,
    ohlcv: pd.DataFrame | None = None,
    features_root: Path | None = None,
    spec: SqueezeLabelSpec = DEFAULT_SPEC,
) -> pd.DataFrame:
    sym = symbol.upper()
    feat = features if features is not None else load_features(sym, root=features_root)
    if feat.empty:
        return pd.DataFrame()

    bars = ohlcv
    if bars is None:
        try:
            bars = fetch_daily_ohlcv(sym)
        except Exception:  # noqa: BLE001
            bars = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, fr in feat.iterrows():
        as_of = str(fr.get("as_of"))[:10]
        candidate = is_candidate_setup(fr.to_dict())
        row: dict[str, Any] = {
            "symbol": sym,
            "as_of": as_of,
            "candidate_setup": int(candidate),
            "positioning_stress": float(fr.get("positioning_stress") or 0.0),
        }
        for h in _LABEL_HORIZONS:
            fwd = _forward_return(bars, as_of, h)
            row[f"fwd_ret_{h}d"] = fwd
            row[f"magnitude_label_{h}d"] = fwd
            if h in HORIZONS:
                squeeze = int(candidate and is_squeeze_outcome(fwd, h, spec=spec))
                # Also label non-candidate large upside as 0 (negative class)
                if not candidate and fwd is not None:
                    squeeze = 0
                row[f"squeeze_{h}d"] = squeeze
                # Duration proxy: first day within horizon that crosses threshold
                row[f"duration_label_{h}d"] = _duration_to_threshold(bars, as_of, h, spec)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        save_labels(sym, df, root=features_root)
    return df


def _forward_return(ohlcv: pd.DataFrame, as_of: str, horizon: int) -> float | None:
    if ohlcv is None or ohlcv.empty or as_of not in ohlcv.index:
        return None
    idx = list(ohlcv.index)
    i = idx.index(as_of)
    j = i + horizon
    if j >= len(idx):
        return None
    c0 = float(ohlcv["close"].iloc[i])
    c1 = float(ohlcv["close"].iloc[j])
    if c0 <= 0:
        return None
    return (c1 / c0) - 1.0


def _duration_to_threshold(
    ohlcv: pd.DataFrame,
    as_of: str,
    horizon: int,
    spec: SqueezeLabelSpec,
) -> float | None:
    if ohlcv is None or ohlcv.empty or as_of not in ohlcv.index:
        return None
    idx = list(ohlcv.index)
    i = idx.index(as_of)
    c0 = float(ohlcv["close"].iloc[i])
    if c0 <= 0:
        return None
    thr = spec.threshold_for(horizon)
    for step in range(1, horizon + 1):
        j = i + step
        if j >= len(idx):
            break
        ret = (float(ohlcv["close"].iloc[j]) / c0) - 1.0
        if ret >= thr:
            return float(step)
    return float(horizon)
