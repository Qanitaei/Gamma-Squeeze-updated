"""Alpaca API client using legacy key authentication.

See: https://docs.alpaca.markets/docs/authentication
"""

from __future__ import annotations

import json
import os
import ssl
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

DATA_API_BASE = "https://data.alpaca.markets"
TRADING_API_LIVE = "https://api.alpaca.markets"
TRADING_API_PAPER = "https://paper-api.alpaca.markets"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def load_alpaca_config() -> dict[str, str]:
    load_dotenv()
    key_id = (
        os.getenv("ALPACA_API_KEY_ID", "").strip()
        or os.getenv("APCA_API_KEY_ID", "").strip()
    )
    secret = (
        os.getenv("ALPACA_API_SECRET_KEY", "").strip()
        or os.getenv("APCA_API_SECRET_KEY", "").strip()
    )
    paper = os.getenv("ALPACA_PAPER", "").strip().lower() in ("1", "true", "yes")
    feed = os.getenv("ALPACA_OPTION_FEED", "").strip() or "indicative"
    data_base = os.getenv("ALPACA_DATA_API_BASE", "").strip() or DATA_API_BASE
    trading_env = os.getenv("ALPACA_TRADING_API_BASE", "").strip()
    if trading_env:
        trading_base = trading_env.rstrip("/")
        if trading_base.endswith("/v2"):
            trading_base = trading_base[:-3]
    else:
        trading_base = TRADING_API_PAPER if paper else TRADING_API_LIVE

    missing = [
        name
        for name, value in (
            ("ALPACA_API_KEY_ID", key_id),
            ("ALPACA_API_SECRET_KEY", secret),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"Missing required Alpaca credentials: {', '.join(missing)}. "
            "Add ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY to .env "
            "(from https://app.alpaca.markets → API Keys)."
        )

    return {
        "key_id": key_id,
        "secret": secret,
        "data_base": data_base.rstrip("/"),
        "trading_base": trading_base.rstrip("/"),
        "option_feed": feed,
    }


class AlpacaClient:
    """Minimal Alpaca REST client with APCA auth headers."""

    def __init__(
        self,
        *,
        key_id: str,
        secret: str,
        data_base: str = DATA_API_BASE,
        trading_base: str = TRADING_API_LIVE,
        option_feed: str = "indicative",
        throttle_s: float = 0.25,
    ) -> None:
        self.key_id = key_id
        self.secret = secret
        self.data_base = data_base.rstrip("/")
        trading = trading_base.rstrip("/")
        if trading.endswith("/v2"):
            trading = trading[:-3]
        self.trading_base = trading
        self.option_feed = option_feed
        self.throttle_s = throttle_s

    @classmethod
    def from_env(cls, *, throttle_s: float = 0.25) -> AlpacaClient:
        cfg = load_alpaca_config()
        return cls(
            key_id=cfg["key_id"],
            secret=cfg["secret"],
            data_base=cfg["data_base"],
            trading_base=cfg["trading_base"],
            option_feed=cfg["option_feed"],
            throttle_s=throttle_s,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret,
            "Accept": "application/json",
        }

    def get(self, base: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = ""
        if params:
            clean = {k: v for k, v in params.items() if v is not None and v != ""}
            if clean:
                query = "?" + urlencode(clean)
        url = f"{base.rstrip('/')}{path}{query}"
        req = Request(url, headers=self._headers(), method="GET")
        try:
            with urlopen(req, timeout=120, context=_ssl_context()) as resp:
                payload = resp.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Alpaca HTTP {exc.code} {path}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Alpaca request failed {path}: {exc}") from exc
        finally:
            if self.throttle_s > 0:
                time.sleep(self.throttle_s)

        if not payload:
            return {}
        return json.loads(payload)

    def get_data(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.get(self.data_base, path, params)

    def get_trading(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.get(self.trading_base, path, params)

    def verify(self) -> dict[str, Any]:
        return self.get_trading("/v2/account")


def create_client(*, throttle_s: float = 0.25) -> AlpacaClient:
    return AlpacaClient.from_env(throttle_s=throttle_s)
