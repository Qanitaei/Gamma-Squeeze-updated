"""Stack connection settings (Python 3.12+)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from gamma_squeeze.config import load_env, resolve_matrix_root
from gamma_squeeze.stack.inventory import PYTHON_REQUIRES


@dataclass(frozen=True)
class StackSettings:
    database_url: str
    redis_url: str
    kafka_bootstrap: str
    duckdb_path: Path
    mlflow_tracking_uri: str
    python_requires: str = PYTHON_REQUIRES
    phase: int = 2

    def to_public_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duckdb_path"] = str(self.duckdb_path)
        return d


@lru_cache(maxsize=1)
def get_stack_settings() -> StackSettings:
    load_env()
    matrix = resolve_matrix_root()
    duck = os.getenv("DUCKDB_PATH", "").strip()
    duck_path = Path(duck).expanduser() if duck else matrix / "analytics" / "gamma.duckdb"
    return StackSettings(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql://gamma:gamma@localhost:5432/gamma_squeeze",
        ),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        kafka_bootstrap=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        duckdb_path=duck_path,
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        python_requires=PYTHON_REQUIRES,
        phase=2,
    )
