"""DuckDB analytical store over local feature / forecast Parquet-friendly paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gamma_squeeze.stack.settings import get_stack_settings


def connect():
    try:
        import duckdb
    except ImportError:
        return None
    path = get_stack_settings().duckdb_path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(path))
    except OSError:
        return None


def health() -> dict[str, Any]:
    path = str(get_stack_settings().duckdb_path)
    try:
        con = connect()
    except Exception as exc:  # noqa: BLE001
        return {"duckdb": "down", "path": path, "error": str(exc)}
    if con is None:
        return {"duckdb": "down", "path": path}
    try:
        con.execute("SELECT 1").fetchone()
        return {"duckdb": "up", "path": path}
    except Exception as exc:  # noqa: BLE001
        return {"duckdb": "down", "path": path, "error": str(exc)}
    finally:
        try:
            con.close()
        except Exception:  # noqa: BLE001
            pass


def register_parquet_glob(table: str, glob_path: str | Path) -> bool:
    con = connect()
    if con is None:
        return False
    try:
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{glob_path}')"
        )
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        con.close()
