"""Model version registry for pluggable backends (baseline / RL / deep)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from gamma_squeeze.config import resolve_models_root


@dataclass
class ModelRecord:
    name: str
    backend: str  # baseline | rl | deep
    version: str
    path: str
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    active: bool = False


_BACKENDS: dict[str, Callable[..., Any]] = {}


def register_backend(name: str, factory: Callable[..., Any]) -> None:
    _BACKENDS[name] = factory


def list_backends() -> list[str]:
    return sorted(_BACKENDS.keys())


def get_backend(name: str) -> Callable[..., Any] | None:
    return _BACKENDS.get(name)


def registry_path(root: Path | None = None) -> Path:
    return (root or resolve_models_root()) / "registry.json"


def load_registry(root: Path | None = None) -> dict[str, Any]:
    path = registry_path(root)
    if not path.is_file():
        return {"models": [], "active": {}}
    with path.open() as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"models": [], "active": {}}


def save_registry(data: dict[str, Any], root: Path | None = None) -> Path:
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    return path


def register_model(
    *,
    name: str,
    backend: str,
    version: str,
    path: str | Path,
    metrics: dict[str, Any] | None = None,
    activate: bool = True,
    root: Path | None = None,
) -> ModelRecord:
    reg = load_registry(root)
    record = ModelRecord(
        name=name,
        backend=backend,
        version=version,
        path=str(path),
        metrics=metrics or {},
        created_at=datetime.now(timezone.utc).isoformat(),
        active=activate,
    )
    models = reg.setdefault("models", [])
    if activate:
        for m in models:
            if m.get("name") == name:
                m["active"] = False
        reg.setdefault("active", {})[name] = version
    models.append(asdict(record))
    save_registry(reg, root)
    return record


def active_version(name: str, root: Path | None = None) -> str | None:
    reg = load_registry(root)
    return (reg.get("active") or {}).get(name)
