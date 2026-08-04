"""Asynchronous HTTP helpers for gateway / service fan-out."""

from __future__ import annotations

from typing import Any

import httpx

from gamma_squeeze.core.settings import load_platform_settings


async def aget_json(url: str, *, timeout: float | None = None) -> dict[str, Any]:
    settings = load_platform_settings()
    t = timeout if timeout is not None else settings.request_timeout_seconds
    async with httpx.AsyncClient(timeout=t) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {"data": data}


async def apost_json(url: str, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
    settings = load_platform_settings()
    t = timeout if timeout is not None else settings.request_timeout_seconds
    async with httpx.AsyncClient(timeout=t) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {"data": data}


async def gather_health(urls: dict[str, str]) -> dict[str, Any]:
    """Concurrent /health checks for registered service URLs."""
    import asyncio

    async def _one(name: str, url: str) -> tuple[str, dict[str, Any]]:
        try:
            body = await aget_json(url, timeout=2.0)
            return name, body
        except Exception as exc:  # noqa: BLE001
            return name, {"status": "down", "error": str(exc), "url": url}

    pairs = await asyncio.gather(*[_one(n, u) for n, u in urls.items()])
    return {n: body for n, body in pairs}
