"""Model version control — semantic versions + registry snapshots."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from gamma_squeeze.config import resolve_models_root
from gamma_squeeze.retrain.registry import load_registry, register_model, save_registry


@dataclass
class ModelVersion:
    name: str
    version: str
    backend: str
    path: str
    created_at: str
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    parent_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _versions_dir(root: Path | None = None) -> Path:
    return (root or resolve_models_root()) / "versions"


def next_version(name: str = "squeeze_probability", *, root: Path | None = None) -> str:
    """Bump patch version based on existing version folders / registry."""
    reg = load_registry(root)
    versions = [
        m.get("version", "")
        for m in reg.get("models") or []
        if m.get("name") == name and str(m.get("version", "")).startswith("v")
    ]
    best = 0
    for v in versions:
        try:
            # v1.2.3 or v12
            parts = v.lstrip("v").split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            best = max(best, major * 1_000_000 + minor * 1000 + patch)
        except ValueError:
            continue
    best += 1
    major, rem = divmod(best, 1_000_000)
    minor, patch = divmod(rem, 1000)
    if major == 0:
        major = 1
    return f"v{major}.{minor}.{patch}"


def save_versioned_model(
    model: Any,
    *,
    name: str,
    backend: str,
    metrics: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    parent_version: str | None = None,
    activate: bool = True,
    root: Path | None = None,
) -> ModelVersion:
    version = next_version(name, root=root)
    base = _versions_dir(root) / name / version
    base.mkdir(parents=True, exist_ok=True)
    model_path = base / "model.joblib"
    meta_path = base / "meta.json"
    joblib.dump(model, model_path)
    record = ModelVersion(
        name=name,
        version=version,
        backend=backend,
        path=str(model_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        metrics=metrics or {},
        tags=tags or {},
        parent_version=parent_version,
    )
    with meta_path.open("w") as f:
        json.dump(record.to_dict(), f, indent=2)
    # latest pointer
    latest = _versions_dir(root) / name / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest, ignore_errors=True)
    try:
        latest.symlink_to(base, target_is_directory=True)
    except OSError:
        # copy meta pointer file if symlink fails
        with ( _versions_dir(root) / name / "latest.json").open("w") as f:
            json.dump({"version": version, "path": str(model_path)}, f)

    register_model(
        name=name,
        backend=backend,
        version=version,
        path=model_path,
        metrics=metrics,
        activate=activate,
        root=root,
    )
    return record


def list_versions(name: str = "squeeze_probability", *, root: Path | None = None) -> list[dict[str, Any]]:
    reg = load_registry(root)
    return [m for m in (reg.get("models") or []) if m.get("name") == name]


def rollback_version(name: str, version: str, *, root: Path | None = None) -> dict[str, Any]:
    reg = load_registry(root)
    target = None
    for m in reg.get("models") or []:
        if m.get("name") == name and m.get("version") == version:
            target = m
            m["active"] = True
        elif m.get("name") == name:
            m["active"] = False
    if not target:
        return {"ok": False, "error": f"version_not_found:{version}"}
    reg.setdefault("active", {})[name] = version
    save_registry(reg, root)
    return {"ok": True, "active": version, "path": target.get("path")}
