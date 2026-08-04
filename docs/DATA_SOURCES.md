# Data Sources (Phase 3)

Adapters live under `src/gamma_squeeze/data_sources/` and are exposed by the
**Data Collection** service (`:8001`). Machine-readable catalog:
`gamma_squeeze.data_sources.catalog.LOCKED_ADAPTERS`.

**Phase status:** locked. See [ARCHITECTURE.md](../ARCHITECTURE.md).

## Historical Options Matrix

| Adapter | ID | Support |
|---------|----|---------|
| **Alpaca** | `options.alpaca` | **Primary** — live Alpaca API → local SSD → Cloudflare KV namespace **`alpaca-options-matrix-backup`** (`ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY`, `ALPACA_BACKUP_WORKER_URL`) |
| **Interactive Brokers** | `options.ibkr` | TWS / IB Gateway — `IBKR_HOST` / `IBKR_PORT` / `IBKR_CLIENT_ID` |
| **OPRA** | `options.opra` | `OPRA_DATA_PATH` or `OPRA_API_KEY` |
| **Polygon** | `options.polygon` | Live snapshot — `POLYGON_API_KEY` |
| **CBOE** | `options.cboe` | Public delayed for VIX/VVIX; equity needs `CBOE_API_KEY` |

## Market Data

| Adapter | ID | Notes |
|---------|----|-------|
| **OHLCV** | `market.ohlcv` | Daily/intraday bars (worker + Alpaca) |
| **Tick Data** | `market.tick` | Tick stream adapter (stub until stream wired) |
| **VWAP** | `market.vwap` | Derived from OHLCV |
| **Order Book** | `market.order_book` | Alpaca top-of-book / optional L2 SSD |
| **Volume Profile** | `market.volume_profile` | Derived from OHLCV |

## Macroeconomic

| Adapter | ID | Notes |
|---------|----|-------|
| **FRED** | `macro.fred` | FRED worker / SSD mirror |
| **Treasury** | `macro.treasury` | DGS2/10/30 + T10Y2Y |
| **Federal Reserve** | `macro.federal_reserve` | Fed funds / policy proxies |
| **Economic Calendar** | `macro.economic_calendar` | Calendar worker / event feed |

Local mirrors: `EXPORT_FRED_ECONOMIC_DATA_SSD_PATH` or `/Volumes/PortableSSD/Fred Economic data`.

## Market Structure

| Adapter | ID | Symbol / coverage |
|---------|----|-------------------|
| **VIX** | `structure.vix` | CBOE VIX |
| **VVIX** | `structure.vvix` | CBOE VVIX |
| **MOVE** | `structure.move` | ICE BofA MOVE |
| **DXY** | `structure.dxy` | US Dollar Index |
| **USDJPY** | `structure.usdjpy` | USD/JPY |
| **Treasury Yields** | `structure.treasury_yields` | Curve via treasury/FRED |

Composites (also registered): `structure.vol_indices`, `structure.fx`, `structure.yields`.

## Corporate

| Adapter | ID | Auth |
|---------|----|------|
| **Earnings** | `corporate.earnings` | `POLYGON_API_KEY` or `FINNHUB_API_KEY` |
| **Dividends** | `corporate.dividends` | `POLYGON_API_KEY` |
| **Splits** | `corporate.splits` | `POLYGON_API_KEY` |

## Calendar

| Adapter | ID | Coverage |
|---------|----|----------|
| **OPEX** | `calendar.opex` | Monthly 3rd-Friday options expiration |
| **Monthly Expiration** | `calendar.monthly_expiration` | Monthly equity OPEX dates |
| **Quarterly Expiration** | `calendar.quarterly_expiration` | Mar/Jun/Sep/Dec + triple witching |
| **Presidential Cycle** | `calendar.presidential_cycle` | 4-year presidential cycle |
| **Midterm Cycle** | `calendar.midterm_cycle` | Midterm election dates |

Composites: `calendar.expiration`, `calendar.political_cycle`.

## API

```bash
# locked + registered catalog
curl http://127.0.0.1:8001/v1/adapters

# health all
curl http://127.0.0.1:8001/v1/adapters/health

# fetch via adapter
curl -X POST http://127.0.0.1:8001/v1/adapters/options.alpaca/fetch \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","as_of":"2026-07-31"}'

curl -X POST http://127.0.0.1:8001/v1/adapters/structure.vix/fetch \
  -H 'content-type: application/json' -d '{}'

curl -X POST http://127.0.0.1:8001/v1/adapters/calendar.monthly_expiration/fetch \
  -H 'content-type: application/json' -d '{"year":2026}'
```

## Local materialization (Alpaca → Portable SSD)

1. Export options matrices (sibling plane or platform scripts) → `/Volumes/PortableSSD/Alpaca ticker data`
2. `scripts/export_gamma_squeeze_matrix_from_kv.py` → `/Volumes/PortableSSD/Gamma Squeeze Matrix`
3. Feature Engineering prefers local matrices over live KV

## Phase 3 complete criteria

- Locked catalog covers Options / Market / Macro / Structure / Corporate / Calendar as specified
- Every locked adapter ID registers and exposes `health()` (+ `fetch` / `fetch_matrix`)
- Data Collection `/v1/adapters` reports `phase: 3` and `locked_ok: true`
- Adapter tests pass without live broker keys
