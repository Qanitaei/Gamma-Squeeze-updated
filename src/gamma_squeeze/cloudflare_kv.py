"""Cloudflare KV credentials + helpers for gamma-squeeze-data exports."""

from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import certifi
from dotenv import dotenv_values, load_dotenv

from gamma_squeeze.config import PLATFORM_ROOT

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
GAMMA_SQUEEZE_DATA_NAMESPACE = "gamma-squeeze-data"
DEFAULT_NAMESPACE_ID = "f949a0301f604312a9c8959f6f2a3918"
CRED_CANDIDATES = [
    Path("/Users/ruslantkach/Desktop/economic-calendar/.cloudflare-credentials.json"),
    PLATFORM_ROOT / ".cloudflare-credentials.json",
]


def clean_api_token(raw: str) -> str:
    """Strip YAML list corruption such as ``\\t- cfat_...``."""
    text = (raw or "").replace("\\t", "\t")
    text = re.sub(r"^[\s\-]+", "", text.strip())
    match = re.search(r"(cfat_[A-Za-z0-9_\-]+)", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-Za-z0-9_\-]{40,})", text)
    return match.group(1) if match else text.strip()


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


def _credential_pairs() -> list[tuple[str, str]]:
    """Candidate (account_id, token) pairs — prefer numbered working secrets."""
    load_cloudflare_env()
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for path in CRED_CANDIDATES:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        account = str(data.get("account_id") or "").strip().lower()
        token = clean_api_token(str(data.get("api_token") or ""))
        if account and token and (account, token) not in seen:
            pairs.append((account, token))
            seen.add((account, token))

    accounts: list[str] = []
    tokens: list[str] = []
    for suffix in ("1", "", "2", "3"):
        acc = os.getenv(f"CLOUDFLARE_ACCOUNT_ID{suffix}", "").strip().lower()
        tok = clean_api_token(os.getenv(f"CLOUDFLARE_API_TOKEN{suffix}", ""))
        if acc and acc not in accounts:
            accounts.append(acc)
        if tok and tok not in tokens:
            tokens.append(tok)

    # Prefer same-suffix pairs first (ACCOUNT_ID1 × TOKEN1), then cross products.
    for suffix in ("1", "", "2", "3"):
        acc = os.getenv(f"CLOUDFLARE_ACCOUNT_ID{suffix}", "").strip().lower()
        tok = clean_api_token(os.getenv(f"CLOUDFLARE_API_TOKEN{suffix}", ""))
        if acc and tok and (acc, tok) not in seen:
            pairs.append((acc, tok))
            seen.add((acc, tok))
    for acc in accounts:
        for tok in tokens:
            if (acc, tok) not in seen:
                pairs.append((acc, tok))
                seen.add((acc, tok))
    return pairs


def _probe_pair(account_id: str, token: str, namespace_id: str) -> bool:
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{namespace_id}/keys?limit=10"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            body = json.loads(resp.read().decode() or "{}")
        return bool(body.get("success"))
    except Exception:  # noqa: BLE001
        return False


def cloudflare_credentials(
    *,
    namespace_id: str | None = None,
) -> tuple[str, str]:
    """Return a working Cloudflare account/token pair for gamma-squeeze-data KV."""
    ns_id = (namespace_id or gamma_squeeze_namespace_id()).strip()
    for account_id, token in _credential_pairs():
        if _probe_pair(account_id, token, ns_id):
            return account_id, token
    raise SystemExit(
        "No working Cloudflare credentials for gamma-squeeze-data "
        "(tried CLOUDFLARE_ACCOUNT_ID/TOKEN and numbered variants)"
    )


def gamma_squeeze_namespace_id() -> str:
    meta = PLATFORM_ROOT / "data" / "cloudflare_gamma_squeeze_data.json"
    if meta.is_file():
        try:
            return str(json.loads(meta.read_text(encoding="utf-8"))["namespace_id"])
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    for key in (
        "GAMMA_SQUEEZE_DATA_NAMESPACE_ID",
        "CLOUDFLARE_SQUEEZE_NAMESPACE_ID",
    ):
        val = os.getenv(key, "").strip()
        if val:
            return val
    return DEFAULT_NAMESPACE_ID


def sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf so payloads are ECMA-404 / Workers JSON.parse safe."""
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):  # NaN or ±Inf
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def dumps_kv_json(obj: Any, *, indent: int | None = None) -> str:
    """Serialize for KV — nulls out non-finite floats; never allow_nan."""
    return json.dumps(
        sanitize_for_json(obj),
        indent=indent,
        default=str,
        allow_nan=False,
    )


def kv_put(
    account_id: str,
    token: str,
    namespace_id: str,
    key: str,
    value: str,
    *,
    max_value: int = 20_000_000,
) -> None:
    # Guard against Python's non-standard NaN/Infinity tokens leaking into Workers.
    if "NaN" in value or "Infinity" in value:
        try:
            value = dumps_kv_json(json.loads(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            value = (
                value.replace(": NaN", ": null")
                .replace(":NaN", ": null")
                .replace(" NaN", " null")
                .replace(": Infinity", ": null")
                .replace(":-Infinity", ": null")
                .replace(": -Infinity", ": null")
            )
    if len(value.encode()) > max_value:
        raise RuntimeError(f"Value too large for key {key}: {len(value.encode())} bytes")
    enc = urllib.parse.quote(key, safe="")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{namespace_id}/values/{enc}"
    )
    req = urllib.request.Request(
        url,
        data=value.encode("utf-8"),
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "text/plain; charset=utf-8",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp:
                body = json.loads(resp.read().decode() or "{}")
            if body and body.get("success") is False:
                raise RuntimeError(body)
            return
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"PUT {key} -> {exc.code}: {exc.read()[:400]}") from exc
        except urllib.error.URLError:
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise


def kv_get(account_id: str, token: str, namespace_id: str, key: str) -> Any | None:
    enc = urllib.parse.quote(key, safe="")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{namespace_id}/values/{enc}"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
            raw = resp.read().decode()
        return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def kv_delete(account_id: str, token: str, namespace_id: str, key: str) -> None:
    enc = urllib.parse.quote(key, safe="")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{namespace_id}/values/{enc}"
    )
    req = urllib.request.Request(
        url, method="DELETE", headers={"Authorization": f"Bearer {token}"}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
                resp.read()
            return
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            if exc.code == 404:
                return
            raise


def kv_list_keys(
    account_id: str,
    token: str,
    namespace_id: str,
    *,
    prefix: str = "",
    limit: int = 1000,
) -> list[str]:
    cursor: str | None = None
    names: list[str] = []
    while True:
        q = f"limit={max(10, min(limit, 1000))}"
        if prefix:
            q += f"&prefix={urllib.parse.quote(prefix)}"
        if cursor:
            q += f"&cursor={urllib.parse.quote(cursor)}"
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
            f"/storage/kv/namespaces/{namespace_id}/keys?{q}"
        )
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp:
            body = json.loads(resp.read().decode())
        for row in body.get("result") or []:
            name = row.get("name") if isinstance(row, dict) else str(row)
            if name:
                names.append(name)
        info = body.get("result_info") or {}
        next_cursor = info.get("cursor")
        if next_cursor and (not info.get("list_complete", True) or len(body.get("result") or []) >= 1000):
            cursor = next_cursor
            continue
        break
    return names
