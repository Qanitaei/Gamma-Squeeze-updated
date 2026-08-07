"""Shared Cloudflare KV credential + namespace helpers for gamma-squeeze-data."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from gamma_squeeze.config import PLATFORM_ROOT, resolve_matrix_root

NAMESPACE_META = PLATFORM_ROOT / "data" / "cloudflare_gamma_squeeze_data.json"
CRED_CANDIDATES = [
    Path("/Users/ruslantkach/Desktop/economic-calendar/.cloudflare-credentials.json"),
    PLATFORM_ROOT / ".cloudflare-credentials.json",
]


def clean_api_token(token: str) -> str:
    """Strip YAML-list junk (e.g. ``\\t- cfat_...``) and extract ``cfat_`` tokens."""
    raw = (token or "").strip()
    raw = re.sub(r"^[\t\s\-]+", "", raw)
    match = re.search(r"(cfat_[A-Za-z0-9_\-]+)", raw)
    return match.group(1) if match else raw


def load_cloudflare_env() -> None:
    for path in (
        Path("/Users/ruslantkach/Desktop/schwab-options-export/.env"),
        PLATFORM_ROOT / ".env",
    ):
        if path.is_file():
            load_dotenv(path, override=False)
    schwab = Path("/Users/ruslantkach/Desktop/schwab-options-export/.env")
    if schwab.is_file():
        for key, value in (dotenv_values(schwab) or {}).items():
            if value and not (os.getenv(key) or "").strip():
                os.environ[key] = value
    # Sandbox shells sometimes drop injected secrets; reload blanks from .env with override.
    env_file = PLATFORM_ROOT / ".env"
    if env_file.is_file():
        values = dotenv_values(env_file) or {}
        for key in (
            "CLOUDFLARE_ACCOUNT_ID",
            "CLOUDFLARE_ACCOUNT_ID1",
            "CLOUDFLARE_API_TOKEN",
            "CLOUDFLARE_API_TOKEN1",
        ):
            val = (values.get(key) or "").strip()
            if val and not (os.getenv(key) or "").strip():
                os.environ[key] = val


def cloudflare_credentials() -> tuple[str, str]:
    """Return ``(account_id, api_token)`` with token cleaning + account lowercasing."""
    load_cloudflare_env()
    for path in CRED_CANDIDATES:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            account = str(data.get("account_id") or "").strip().lower()
            token = clean_api_token(str(data.get("api_token") or ""))
            if account and token:
                return account, token

    # Prefer primary env, then legacy *1 aliases (TOKEN1 may be YAML-corrupted).
    candidates = [
        (
            (os.getenv("CLOUDFLARE_ACCOUNT_ID") or "").strip().lower(),
            clean_api_token(os.getenv("CLOUDFLARE_API_TOKEN") or ""),
        ),
        (
            (os.getenv("CLOUDFLARE_ACCOUNT_ID1") or "").strip().lower(),
            clean_api_token(os.getenv("CLOUDFLARE_API_TOKEN1") or ""),
        ),
        (
            (os.getenv("CLOUDFLARE_ACCOUNT_ID1") or "").strip().lower(),
            clean_api_token(os.getenv("CLOUDFLARE_API_TOKEN") or ""),
        ),
        (
            (os.getenv("CLOUDFLARE_ACCOUNT_ID") or "").strip().lower(),
            clean_api_token(os.getenv("CLOUDFLARE_API_TOKEN1") or ""),
        ),
    ]
    for account, token in candidates:
        if account and token and len(account) >= 20:
            return account, token
    raise SystemExit("Missing Cloudflare credentials (CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN)")


def gamma_squeeze_namespace_id() -> str:
    if NAMESPACE_META.is_file():
        return str(json.loads(NAMESPACE_META.read_text(encoding="utf-8"))["namespace_id"])
    for key in ("GAMMA_SQUEEZE_DATA_NAMESPACE_ID", "CLOUDFLARE_SQUEEZE_NAMESPACE_ID"):
        val = os.getenv(key, "").strip()
        if val:
            return val
    raise SystemExit("Missing gamma-squeeze-data namespace id")


def phase14_scan_dir() -> Path:
    """Local phase-14 export dir (PortableSSD or cloud fallback)."""
    return resolve_matrix_root() / "scans" / "phase14_pipeline"
