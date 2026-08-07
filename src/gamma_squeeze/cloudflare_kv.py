"""Shared Cloudflare KV credential + namespace helpers for gamma-squeeze-data."""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import certifi
from dotenv import dotenv_values, load_dotenv

from gamma_squeeze.config import PLATFORM_ROOT, resolve_matrix_root

NAMESPACE_META = PLATFORM_ROOT / "data" / "cloudflare_gamma_squeeze_data.json"
CRED_CANDIDATES = [
    Path("/Users/ruslantkach/Desktop/economic-calendar/.cloudflare-credentials.json"),
    PLATFORM_ROOT / ".cloudflare-credentials.json",
]
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
_PROBE_NS = "f949a0301f604312a9c8959f6f2a3918"  # gamma-squeeze-data


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
    # Sandbox shells sometimes drop injected secrets; fill blanks from .env.
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


def _probe_pair(account_id: str, token: str, namespace_id: str = _PROBE_NS) -> bool:
    """True when account/token can read the gamma-squeeze-data namespace."""
    if not account_id or not token or len(account_id) < 20:
        return False
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{namespace_id}"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            body = json.loads(resp.read().decode() or "{}")
        return bool(body.get("success"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False


def cloudflare_credentials() -> tuple[str, str]:
    """Return ``(account_id, api_token)`` after probing working account/token pairs.

    Injected sandbox env may provide an invalid ``CLOUDFLARE_ACCOUNT_ID`` alongside a
    valid ``CLOUDFLARE_API_TOKEN1``; probe combinations instead of trusting primary names.
    """
    load_cloudflare_env()
    pairs: list[tuple[str, str]] = []
    for path in CRED_CANDIDATES:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            account = str(data.get("account_id") or "").strip().lower()
            token = clean_api_token(str(data.get("api_token") or ""))
            if account and token:
                pairs.append((account, token))

    accounts = [
        (os.getenv("CLOUDFLARE_ACCOUNT_ID1") or "").strip().lower(),
        (os.getenv("CLOUDFLARE_ACCOUNT_ID") or "").strip().lower(),
    ]
    tokens = [
        clean_api_token(os.getenv("CLOUDFLARE_API_TOKEN1") or ""),
        clean_api_token(os.getenv("CLOUDFLARE_API_TOKEN") or ""),
    ]
    # Prefer cfat_ account-API tokens + ACCOUNT_ID1 (known-good in this automation).
    for account in accounts:
        for token in tokens:
            if account and token:
                pairs.append((account, token))

    seen: set[tuple[str, str]] = set()
    for account, token in pairs:
        key = (account, token)
        if key in seen:
            continue
        seen.add(key)
        if _probe_pair(account, token):
            # Publish winning pair as primary for child processes / wrangler.
            os.environ["CLOUDFLARE_ACCOUNT_ID"] = account
            os.environ["CLOUDFLARE_API_TOKEN"] = token
            return account, token

    raise SystemExit(
        "No working Cloudflare credentials for gamma-squeeze-data "
        "(tried CLOUDFLARE_ACCOUNT_ID/ID1 × TOKEN/TOKEN1)"
    )


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
