#!/usr/bin/env python3
"""Create/bind KV namespace gamma-squeeze-data and deploy OpenAPI 3.0 worker."""

from __future__ import annotations

import json
import os
import ssl
import sys
import uuid
import urllib.error
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gamma_squeeze.cloudflare_kv import cloudflare_credentials  # noqa: E402

WORKER_DIR = ROOT / "cloudflare-worker-squeeze"
SCRIPT_NAME = os.getenv(
    "GAMMA_SQUEEZE_DATA_WORKER_NAME",
    "gamma-squeeze-data-icy-shadow-db40",
)
NAMESPACE_NAME = os.getenv("GAMMA_SQUEEZE_DATA_NAMESPACE", "gamma-squeeze-data")
WORKER_URL = os.getenv(
    "GAMMA_SQUEEZE_DATA_WORKER_URL",
    f"https://{SCRIPT_NAME}.2s6m8rz8fc.workers.dev",
)
BINDING = "KV"
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def _credentials() -> tuple[str, str]:
    return cloudflare_credentials()


def _api(
    account_id: str,
    token: str,
    method: str,
    path: str,
    *,
    data: bytes | None = None,
    content_type: str | None = None,
) -> dict:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}{path}"
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} -> {exc.code}: {detail[:800]}") from exc
    if not body.get("success"):
        raise RuntimeError(body)
    return body


def ensure_namespace(account_id: str, token: str) -> str:
    """Return namespace id for gamma-squeeze-data (create if missing)."""
    env_id = (
        os.getenv("CLOUDFLARE_SQUEEZE_NAMESPACE_ID")
        or os.getenv("GAMMA_SQUEEZE_DATA_NAMESPACE_ID")
        or ""
    ).strip()
    listed = _api(account_id, token, "GET", "/storage/kv/namespaces?per_page=100")
    by_title = {n.get("title"): n.get("id") for n in (listed.get("result") or [])}
    # Accept double-dash typo alias
    for title in (NAMESPACE_NAME, "gamma--squeeze-data", "gamma-squeeze-forecasts"):
        if title in by_title and by_title[title]:
            ns_id = str(by_title[title])
            print(f"Using existing KV namespace {title} id={ns_id}")
            return ns_id
    if env_id and env_id in set(by_title.values()):
        print(f"Using CLOUDFLARE_SQUEEZE_NAMESPACE_ID={env_id}")
        return env_id

    created = _api(
        account_id,
        token,
        "POST",
        "/storage/kv/namespaces",
        data=json.dumps({"title": NAMESPACE_NAME}).encode(),
        content_type="application/json",
    )
    ns_id = str((created.get("result") or {}).get("id") or "")
    if not ns_id:
        raise RuntimeError(f"Failed creating namespace: {created}")
    print(f"Created KV namespace {NAMESPACE_NAME} id={ns_id}")
    return ns_id


def _bundle_worker() -> str:
    index_src = (WORKER_DIR / "src" / "index.js").read_text(encoding="utf-8")
    openapi = json.loads((WORKER_DIR / "openapi.json").read_text(encoding="utf-8"))
    openapi["servers"] = [{"url": WORKER_URL, "description": "Production Workers KV"}]
    openapi["info"]["contact"] = {"name": SCRIPT_NAME, "url": WORKER_URL}
    # Inline OpenAPI (Workers modules can't always import JSON without bundler)
    return index_src.replace(
        'import OPENAPI from "../openapi.json";\n',
        f"const OPENAPI = {json.dumps(openapi, separators=(',', ':'))};\n",
    )


def deploy(namespace_id: str, account_id: str, token: str) -> None:
    bundled = _bundle_worker()
    metadata = {
        "main_module": "index.js",
        "compatibility_date": "2024-06-01",
        "bindings": [
            {
                "type": "kv_namespace",
                "name": BINDING,
                "namespace_id": namespace_id,
            }
        ],
        "vars": {
            "WORKER_URL": WORKER_URL,
            "NAMESPACE_NAME": NAMESPACE_NAME,
        },
    }
    boundary = f"----FormBoundary{uuid.uuid4().hex}"
    parts: list[bytes] = []

    def add(name: str, content: str, content_type: str, filename: str = "") -> None:
        disp = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            disp += f'; filename="{filename}"'
        parts.append(
            (
                f"--{boundary}\r\n{disp}\r\nContent-Type: {content_type}\r\n\r\n"
                f"{content}\r\n"
            ).encode()
        )

    add("metadata", json.dumps(metadata), "application/json")
    add("index.js", bundled, "application/javascript+module", "index.js")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/workers/scripts/{SCRIPT_NAME}"
    )
    req = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=180, context=_SSL_CTX) as resp:
        result = json.loads(resp.read().decode())
    if not result.get("success"):
        raise RuntimeError(result)
    etag = (result.get("result") or {}).get("etag", "?")
    print(f"Deployed worker {SCRIPT_NAME} etag={etag}")
    print(f"Bound {BINDING} -> {NAMESPACE_NAME} ({namespace_id})")
    print(f"Worker URL: {WORKER_URL}")

    # Ensure workers.dev subdomain is enabled
    try:
        _api(
            account_id,
            token,
            "POST",
            f"/workers/scripts/{SCRIPT_NAME}/subdomain",
            data=json.dumps({"enabled": True}).encode(),
            content_type="application/json",
        )
        print("workers.dev subdomain enabled")
    except Exception as exc:  # noqa: BLE001
        print(f"subdomain note: {exc}")


def write_wrangler_id(namespace_id: str) -> None:
    path = WORKER_DIR / "wrangler.toml"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        'id = "00000000000000000000000000000000"',
        f'id = "{namespace_id}"',
        1,
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    account_id, token = _credentials()
    print(f"account_id={account_id[:6]}… worker={SCRIPT_NAME}")
    ns_id = ensure_namespace(account_id, token)
    write_wrangler_id(ns_id)
    deploy(ns_id, account_id, token)
    # persist id for upload script
    out = ROOT / "data" / "cloudflare_gamma_squeeze_data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "namespace": NAMESPACE_NAME,
                "namespace_id": ns_id,
                "binding": BINDING,
                "worker": SCRIPT_NAME,
                "worker_url": WORKER_URL,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
