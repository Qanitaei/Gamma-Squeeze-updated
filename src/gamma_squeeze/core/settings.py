"""Configuration via environment variables and YAML files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from gamma_squeeze.config import PLATFORM_ROOT, load_env


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        # Minimal fallback parser for flat-ish YAML we ship (no anchors)
        return _parse_simple_yaml(path.read_text())
    data = yaml.safe_load(path.read_text()) or {}
    return data if isinstance(data, dict) else {}


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Tiny indented-YAML subset when PyYAML is unavailable."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, val = raw.lstrip().partition(":")
        key = key.strip()
        val = val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not val:
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
            continue
        if val.startswith('"') and val.endswith('"'):
            parent[key] = val[1:-1]
        elif val.lower() in ("true", "false"):
            parent[key] = val.lower() == "true"
        else:
            try:
                parent[key] = int(val) if "." not in val else float(val)
            except ValueError:
                parent[key] = val
    return root


def resolve_config_path(explicit: str | Path | None = None) -> Path:
    load_env()
    if explicit:
        return Path(explicit).expanduser()
    env = os.getenv("PLATFORM_CONFIG_PATH", "").strip()
    if env:
        return Path(env).expanduser()
    return PLATFORM_ROOT / "config" / "platform.yaml"


@dataclass(frozen=True)
class PlatformSettings:
    """Typed platform settings merged from YAML + environment."""

    name: str = "gamma-squeeze-platform"
    version: str = "0.2.0"
    python_requires: str = ">=3.12"
    market_tz: str = "America/New_York"
    global_seed: int = 42
    deterministic_preprocess: bool = True
    log_level: str = "INFO"
    log_json: bool = True
    metrics_enabled: bool = True
    metrics_namespace: str = "gamma_squeeze"
    async_preferred: bool = True
    request_timeout_seconds: float = 30.0
    model_backends: dict[str, str] = field(default_factory=dict)
    training_defaults: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def backend(self, key: str, default: str = "") -> str:
        return str(self.model_backends.get(key, default))


def _from_merged(data: dict[str, Any]) -> PlatformSettings:
    plat = data.get("platform") or {}
    repro = data.get("reproducibility") or {}
    logging_cfg = data.get("logging") or {}
    metrics_cfg = data.get("metrics") or {}
    api = data.get("api") or {}
    models = data.get("models") or {}
    training = data.get("training") or {}
    return PlatformSettings(
        name=str(plat.get("name") or "gamma-squeeze-platform"),
        version=str(plat.get("version") or "0.2.0"),
        python_requires=str(plat.get("python_requires") or ">=3.12"),
        market_tz=str(
            os.getenv("FORECAST_MARKET_TZ") or plat.get("market_tz") or "America/New_York"
        ),
        global_seed=int(os.getenv("GLOBAL_SEED") or repro.get("global_seed") or 42),
        deterministic_preprocess=bool(repro.get("deterministic_preprocess", True)),
        log_level=str(os.getenv("LOG_LEVEL") or logging_cfg.get("level") or "INFO"),
        log_json=str(os.getenv("LOG_JSON", str(logging_cfg.get("json", True)))).lower()
        in ("1", "true", "yes"),
        metrics_enabled=str(os.getenv("METRICS_ENABLED", str(metrics_cfg.get("enabled", True)))).lower()
        in ("1", "true", "yes"),
        metrics_namespace=str(metrics_cfg.get("namespace") or "gamma_squeeze"),
        async_preferred=bool(api.get("async_preferred", True)),
        request_timeout_seconds=float(
            os.getenv("REQUEST_TIMEOUT_SECONDS") or api.get("request_timeout_seconds") or 30.0
        ),
        model_backends={str(k): str(v) for k, v in models.items()},
        training_defaults=dict(training),
        raw=data,
    )


@lru_cache(maxsize=4)
def load_platform_settings(config_path: str | None = None) -> PlatformSettings:
    """Load YAML defaults, then apply environment overrides."""
    load_env()
    path = resolve_config_path(config_path)
    yaml_data = load_yaml_file(path)
    return _from_merged(yaml_data)


def clear_settings_cache() -> None:
    load_platform_settings.cache_clear()
