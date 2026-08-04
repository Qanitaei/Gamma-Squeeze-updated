"""Build Alpaca option-chain rows / matrices (greeks, OI, volume)."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

from gamma_squeeze.ingest.alpaca_client import AlpacaClient

MATRIX_COLUMNS = [
    "underlying_symbol",
    "option_symbol",
    "expiration_date",
    "days_to_expiration",
    "strike_price",
    "put_call",
    "bid",
    "ask",
    "last",
    "mark",
    "open_price",
    "close_price",
    "high_price",
    "low_price",
    "total_volume",
    "open_interest",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "volatility",
    "in_the_money",
    "underlying_price",
    "quote_time_ms",
]

_OCC_RE = re.compile(
    r"^(?P<root>[A-Z]{1,6})\s*(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$"
)


def _group_rows_by_expiration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group flat contract rows into expiration -> strike -> call/put."""
    exp_map: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        exp = str(row.get("expiration_date", "")).split(":")[0]
        strike = str(row.get("strike_price", ""))
        side = (row.get("put_call") or "").upper()
        exp_map.setdefault(exp, {})
        exp_map[exp].setdefault(strike, {"strike_price": row.get("strike_price")})
        if side == "CALL":
            exp_map[exp][strike]["call"] = row
        elif side == "PUT":
            exp_map[exp][strike]["put"] = row

    grouped: list[dict[str, Any]] = []
    for exp_date in sorted(exp_map.keys()):
        strikes = exp_map[exp_date]
        strike_rows = []
        for strike_key in sorted(strikes.keys(), key=lambda x: float(x) if x else 0):
            strike_rows.append(strikes[strike_key])
        dte = None
        for s in strike_rows:
            for side in ("call", "put"):
                if side in s and s[side].get("days_to_expiration") is not None:
                    dte = s[side]["days_to_expiration"]
                    break
            if dte is not None:
                break
        grouped.append(
            {
                "expiration_date": exp_date,
                "days_to_expiration": dte,
                "strikes": strike_rows,
            }
        )
    return grouped


def _parse_occ_symbol(symbol: str) -> dict[str, Any]:
    m = _OCC_RE.match(symbol.strip().upper())
    if not m:
        return {}
    strike = int(m.group("strike")) / 1000.0
    exp = f"20{m.group('yy')}-{m.group('mm')}-{m.group('dd')}"
    return {
        "root_symbol": m.group("root").strip(),
        "expiration_date": exp,
        "put_call": "CALL" if m.group("cp") == "C" else "PUT",
        "strike_price": strike,
    }


def _days_to_expiration(
    expiration_date: str,
    *,
    today: date | None = None,
    as_of_date: date | None = None,
) -> int | None:
    try:
        exp = datetime.strptime(expiration_date[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    ref = as_of_date or today or datetime.now(timezone.utc).date()
    return (exp - ref).days


def _bar_session_date(bar: dict[str, Any] | None) -> str | None:
    if not bar:
        return None
    ts = bar.get("t")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def fetch_underlying_close(client: AlpacaClient, symbol: str, as_of_date: date) -> float | None:
    payload = client.get_data(
        "/v2/stocks/bars",
        {
            "symbols": symbol.upper(),
            "timeframe": "1Day",
            "start": as_of_date.isoformat(),
            "end": as_of_date.isoformat(),
            "limit": 1,
            "feed": "iex",
        },
    )
    bars = (payload.get("bars") or {}).get(symbol.upper()) or []
    if not bars:
        return None
    close = bars[-1].get("c")
    return float(close) if close is not None else None


def fetch_underlying_price(
    client: AlpacaClient,
    symbol: str,
    *,
    as_of_date: date | None = None,
) -> float | None:
    if as_of_date:
        return fetch_underlying_close(client, symbol, as_of_date)
    payload = client.get_data(f"/v2/stocks/{symbol.upper()}/snapshot")
    trade = payload.get("latestTrade") or {}
    quote = payload.get("latestQuote") or {}
    daily = payload.get("dailyBar") or {}
    for candidate in (
        trade.get("p"),
        quote.get("ap"),
        quote.get("bp"),
        daily.get("c"),
    ):
        if candidate is not None:
            return float(candidate)
    return None


def _mark_price(bid: float | None, ask: float | None, last: float | None) -> float | None:
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return round((bid + ask) / 2.0, 4)
    if last is not None:
        return last
    if bid is not None and bid > 0:
        return bid
    if ask is not None and ask > 0:
        return ask
    return None


def fetch_option_bars_eod(
    client: AlpacaClient,
    symbols: list[str],
    as_of_date: date,
) -> dict[str, dict[str, Any]]:
    """Fetch 1Day option bars for market close on *as_of_date* (batched)."""
    out: dict[str, dict[str, Any]] = {}
    if not symbols:
        return out

    chunk_size = 100
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(chunk),
                "timeframe": "1Day",
                "start": as_of_date.isoformat(),
                "end": as_of_date.isoformat(),
                "limit": 10000,
            }
            if page_token:
                params["page_token"] = page_token
            try:
                payload = client.get_data("/v1beta1/options/bars", params)
            except RuntimeError as exc:
                msg = str(exc)
                if "403" in msg or "OPRA" in msg:
                    return out
                raise
            bars_map = payload.get("bars") or {}
            for occ, bars in bars_map.items():
                if bars:
                    out[str(occ).upper()] = bars[-1]
            page_token = payload.get("next_page_token")
            if not page_token:
                break
    return out


def fetch_option_contracts(
    client: AlpacaClient,
    underlying: str,
    *,
    expiration_date_lte: str | None = None,
    expiration_date_gte: str | None = None,
    statuses: tuple[str, ...] = ("active",),
) -> list[dict[str, Any]]:
    """Paginate Trading API /v2/options/contracts for one underlying."""
    underlying = underlying.upper()
    if not expiration_date_lte:
        expiration_date_lte = (date.today() + timedelta(days=730)).isoformat()

    contracts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for status in statuses:
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "underlying_symbols": underlying,
                "status": status,
                "limit": 10000,
                "expiration_date_lte": expiration_date_lte,
            }
            if expiration_date_gte:
                params["expiration_date_gte"] = expiration_date_gte
            if page_token:
                params["page_token"] = page_token
            payload = client.get_trading("/v2/options/contracts", params)
            batch = payload.get("option_contracts") or []
            for contract in batch:
                occ = str(contract.get("symbol") or "").upper()
                if not occ or occ in seen:
                    continue
                seen.add(occ)
                contracts.append(contract)
            page_token = payload.get("next_page_token")
            if not page_token:
                break
    return contracts


def fetch_option_snapshots(client: AlpacaClient, underlying: str) -> dict[str, dict[str, Any]]:
    """Paginate Market Data option chain snapshots for one underlying."""
    underlying = underlying.upper()
    snapshots: dict[str, dict[str, Any]] = {}
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {
            "feed": client.option_feed,
            "limit": 1000,
        }
        if page_token:
            params["page_token"] = page_token
        payload = client.get_data(
            f"/v1beta1/options/snapshots/{underlying}",
            params,
        )
        batch = payload.get("snapshots") or {}
        snapshots.update(batch)
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return snapshots


def _contract_to_row(
    contract: dict[str, Any],
    snapshot: dict[str, Any] | None,
    *,
    underlying: str,
    underlying_price: float | None,
    as_of_date: date | None = None,
    eod_bar: dict[str, Any] | None = None,
) -> dict[str, Any]:
    symbol = str(contract.get("symbol") or "").upper()
    parsed = _parse_occ_symbol(symbol)
    exp_date = str(contract.get("expiration_date") or parsed.get("expiration_date") or "")
    strike = contract.get("strike_price")
    try:
        strike_f = float(strike) if strike is not None else parsed.get("strike_price")
    except (TypeError, ValueError):
        strike_f = parsed.get("strike_price")

    opt_type = str(contract.get("type") or "").lower()
    put_call = "CALL" if opt_type == "call" else "PUT" if opt_type == "put" else parsed.get("put_call", "")

    quote = (snapshot or {}).get("latestQuote") or {}
    trade = (snapshot or {}).get("latestTrade") or {}
    daily = (snapshot or {}).get("dailyBar") or {}
    prev_daily = (snapshot or {}).get("prevDailyBar") or {}
    greeks = (snapshot or {}).get("greeks") or {}

    as_of_label = as_of_date.isoformat() if as_of_date else None
    bar = eod_bar
    if bar is None and as_of_label and _bar_session_date(daily) == as_of_label:
        bar = daily

    close_raw = None
    if bar is not None:
        close_raw = bar.get("c")
    elif as_of_label and str(contract.get("close_price_date") or "")[:10] == as_of_label:
        close_raw = contract.get("close_price")
    elif as_of_date is None:
        close_raw = daily.get("c") or prev_daily.get("c")

    try:
        close_price = float(close_raw) if close_raw not in (None, "") else None
    except (TypeError, ValueError):
        close_price = None

    if as_of_date is not None:
        bid = None
        ask = None
        last = close_price
        mark = close_price
        open_price = bar.get("o") if bar else None
        high_price = bar.get("h") if bar else None
        low_price = bar.get("l") if bar else None
        vol = bar.get("v") if bar else None
        quote_ts = bar.get("t") if bar else f"{as_of_label}T20:00:00-04:00"
        use_greeks = as_of_label is None or _bar_session_date(daily) == as_of_label
        if not use_greeks:
            greeks = {}
    else:
        bid = quote.get("bp")
        ask = quote.get("ap")
        last = trade.get("p")
        mark = _mark_price(
            float(bid) if bid is not None else None,
            float(ask) if ask is not None else None,
            float(last) if last is not None else None,
        )
        open_price = daily.get("o")
        high_price = daily.get("h")
        low_price = daily.get("l")
        vol = daily.get("v")
        close_price = daily.get("c") or prev_daily.get("c")
        quote_ts = quote.get("t") or trade.get("t")

    oi_raw = contract.get("open_interest")
    try:
        open_interest = int(float(oi_raw)) if oi_raw not in (None, "") else None
    except (TypeError, ValueError):
        open_interest = None

    dte = _days_to_expiration(exp_date, as_of_date=as_of_date)
    itm = None
    if underlying_price is not None and strike_f is not None and put_call:
        if put_call == "CALL":
            itm = underlying_price > float(strike_f)
        elif put_call == "PUT":
            itm = underlying_price < float(strike_f)

    quote_time_ms = None
    if quote_ts:
        try:
            quote_time_ms = int(
                datetime.fromisoformat(str(quote_ts).replace("Z", "+00:00")).timestamp() * 1000
            )
        except ValueError:
            quote_time_ms = None

    return {
        "underlying_symbol": underlying,
        "option_symbol": symbol,
        "expiration_date": exp_date,
        "days_to_expiration": dte,
        "strike_price": strike_f,
        "put_call": put_call,
        "bid": bid,
        "ask": ask,
        "last": last,
        "mark": mark,
        "open_price": open_price,
        "close_price": close_price,
        "high_price": high_price,
        "low_price": low_price,
        "total_volume": int(vol) if vol is not None else None,
        "open_interest": open_interest,
        "delta": greeks.get("delta"),
        "gamma": greeks.get("gamma"),
        "theta": greeks.get("theta"),
        "vega": greeks.get("vega"),
        "rho": greeks.get("rho"),
        "volatility": (snapshot or {}).get("impliedVolatility"),
        "in_the_money": itm,
        "underlying_price": underlying_price,
        "quote_time_ms": quote_time_ms,
    }


def build_options_rows(
    client: AlpacaClient,
    symbol: str,
    *,
    as_of_date: date | None = None,
    max_dte_days: int | None = None,
    historical_require_activity: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    underlying = symbol.upper()
    underlying_price = fetch_underlying_price(client, underlying, as_of_date=as_of_date)
    today = datetime.now(timezone.utc).date()
    historical = as_of_date is not None and as_of_date < today
    if max_dte_days is not None and as_of_date is not None:
        expiration_date_lte = (as_of_date + timedelta(days=int(max_dte_days))).isoformat()
    elif as_of_date is not None:
        expiration_date_lte = (as_of_date + timedelta(days=730)).isoformat()
    elif max_dte_days is not None:
        expiration_date_lte = (today + timedelta(days=int(max_dte_days))).isoformat()
    else:
        expiration_date_lte = None
    contracts = fetch_option_contracts(
        client,
        underlying,
        expiration_date_gte=as_of_date.isoformat() if as_of_date else None,
        expiration_date_lte=expiration_date_lte,
        statuses=("active", "inactive") if historical else ("active",),
    )
    snapshots: dict[str, dict[str, Any]] = {}
    if not historical:
        snapshots = fetch_option_snapshots(client, underlying)

    eod_bars: dict[str, dict[str, Any]] = {}
    if as_of_date is not None:
        occ_symbols = [
            str(c.get("symbol") or "").upper()
            for c in contracts
            if c.get("symbol") and _OCC_RE.match(str(c.get("symbol") or "").upper())
        ]
        eod_bars = fetch_option_bars_eod(client, occ_symbols, as_of_date)

    rows: list[dict[str, Any]] = []
    expirations: set[str] = set()
    for contract in contracts:
        occ = str(contract.get("symbol") or "").upper()
        row = _contract_to_row(
            contract,
            snapshots.get(occ),
            underlying=underlying,
            underlying_price=underlying_price,
            as_of_date=as_of_date,
            eod_bar=eod_bars.get(occ),
        )
        if max_dte_days is not None:
            dte = row.get("days_to_expiration")
            if dte is not None and int(dte) > int(max_dte_days):
                continue
        if historical_require_activity and historical:
            oi = row.get("open_interest") or 0
            vol = row.get("total_volume") or 0
            if not oi and not vol:
                continue
        rows.append(row)
        exp = str(row.get("expiration_date") or "")
        if exp:
            expirations.add(exp[:10])

    meta = {
        "underlying": underlying,
        "underlying_price": underlying_price,
        "contracts": len(rows),
        "expirations_fetched": len(expirations),
        "snapshot_count": len(snapshots),
        "contract_count": len(contracts),
        "as_of_date": as_of_date.isoformat() if as_of_date else None,
        "session": "market_close" if as_of_date else "latest",
        "market_timezone": "America/New_York" if as_of_date else None,
        "eod_bars_matched": sum(1 for occ in eod_bars if occ in {r["option_symbol"] for r in rows}),
        "max_dte_days": max_dte_days,
    }
    return rows, meta
