"""Polars / Arrow helpers for feature tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def to_polars(df: pd.DataFrame):
    try:
        import polars as pl
    except ImportError:
        return None
    return pl.from_pandas(df)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import polars as pl

        pl.from_pandas(df).write_parquet(path)
    except Exception:  # noqa: BLE001
        df.to_parquet(path, index=False)
    return path


def health() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import polars as pl

        out["polars"] = pl.__version__
    except ImportError:
        out["polars"] = "not_installed"
    try:
        import pyarrow as pa

        out["pyarrow"] = pa.__version__
    except ImportError:
        out["pyarrow"] = "not_installed"
    return out
