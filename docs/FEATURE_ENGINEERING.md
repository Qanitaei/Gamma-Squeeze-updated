# Feature Engineering Engine

Service: Feature Engineering `:8002`

**Options matrices** for dealer features come from **Alpaca** only:
live Alpaca API → local SSD → Cloudflare KV namespace **`alpaca-options-matrix-backup`**
(via `options.alpaca`). Schwab is not used.

## Dealer Positioning (complete)

Module: `src/gamma_squeeze/features/dealer_positioning.py`  
Endpoint: `POST|GET /v1/dealer-positioning/{symbol}`  
Catalog constant: `DEALER_POSITIONING_FIELDS`

## Calculated fields

| Feature | Description |
|---------|-------------|
| `net_gex` | Call GEX − Put GEX (`γ · OI · 100 · S`) |
| `gamma_exposure` | Absolute net GEX magnitude |
| `gamma_by_strike` | Per-strike signed GEX profile |
| `gamma_flip` / `zero_gamma` | Spot where cumulative strike GEX crosses 0 |
| `call_wall` | Strike with max positive GEX |
| `put_wall` | Strike with max negative GEX |
| `dealer_hedge_requirement` | Dealer share hedge (−Σ δ · OI · 100) |
| `dealer_share_imbalance` | (call_oi − put_oi) / total_oi |
| `dealer_delta` | Aggregated dealer delta |
| `dealer_gamma` | Aggregated dealer gamma |
| `dealer_charm` | Σ dδ/dτ · OI · 100 |
| `dealer_vanna` | Σ dδ/dσ · OI · 100 |
| `dealer_vomma` / `dealer_volga` | Σ dν/dσ · OI · 100 |
| `dealer_vega` | Aggregated vega |
| `dealer_theta` | Aggregated theta |
| `dealer_speed` | Σ dγ/dS · OI · 100 |
| `dealer_color` | Σ dγ/dτ · OI · 100 |
| `dealer_zomma` | Σ dγ/dσ · OI · 100 |
| `dealer_ultima` | Σ d(vomma)/dσ · OI · 100 |
| `dealer_cross_gamma` | Moneyness-weighted gamma |
| `dealer_elasticity` | Aggregate omega-style elasticity |
| `dealer_convexity` | Aggregate γ · S² |
| `dealer_liquidity_score` | Volume/OI liquidity proxy ∈ [0,1] |
| `dealer_inventory_estimate` | −hedge requirement (inventory view) |
| `dealer_gamma_acceleration` | Δ dealer_gamma vs prior session |
| `dealer_hedging_velocity` | d(hedge)/dS or Δ hedge vs prior |

Convention: **dealers short customer-long options** (dealer greek = −customer greek × OI × 100).

## Options Metrics (complete)

Module: `src/gamma_squeeze/features/options_metrics.py`  
Endpoint: `POST|GET /v1/options-metrics/{symbol}`  
Catalog: `OPTIONS_METRICS_FIELDS`  
Data: Alpaca API → SSD → KV **`alpaca-options-matrix-backup`**

| Feature | Description |
|---------|-------------|
| `open_interest` | Total call+put OI |
| `oi_velocity` | ΔOI / prior OI |
| `oi_delta` | Absolute ΔOI vs prior session |
| `volume_oi_ratio` | Total volume / total OI |
| `call_volume` / `put_volume` | Side volumes |
| `put_call_ratio` | Put OI / Call OI |
| `skew` | ~25Δ put IV − call IV (nearest expiry) |
| `smile` | Wing avg IV − ATM IV |
| `term_structure` | Far ATM IV − near ATM IV |
| `iv_rank` | (IV − min) / (max − min) over history |
| `iv_percentile` | Fraction of history with IV < current |
| `historical_volatility` | ~252d close-to-close HV |
| `realized_volatility` | ~20d annualized RV |
| `forward_volatility` | Calendar forward vol between first two expiries |
| `expected_move` | 1σ dollar move to nearest expiry (`expected_move_pct` also returned) |
| `probability_itm` | RN P(ITM) for ATM call |
| `probability_touch` | Approx. barrier touch prob to +1σ |
| `risk_neutral_density` | Breeden-Litzenberger density curve |

Also returns `atm_iv`, `smile_curve`, and `term_curve` when `include_curves=true`.

## Technical Indicators (complete)

Module: `src/gamma_squeeze/features/technical_indicators.py`  
Endpoint: `POST|GET /v1/technical-indicators/{symbol}`  
Catalog: `TECHNICAL_INDICATORS_FIELDS`  
Data: OHLCV via Alpaca / OHLCV worker

| Feature | Description |
|---------|-------------|
| `ema` | EMA 12 / 26 / 50 |
| `sma` | SMA 20 / 50 / 200 |
| `vwap` | Cumulative typical-price VWAP |
| `anchored_vwap` | VWAP from `anchor_date` (or last 60 bars) |
| `atr` | 14-period average true range |
| `rsi` | 14-period RSI |
| `macd` | MACD(12,26,9) line / signal / hist |
| `adx` | ADX + +DI / −DI |
| `cci` | 20-period Commodity Channel Index |
| `keltner` | Keltner upper / lower (EMA ± 1.5 ATR) |
| `donchian` | 20-period Donchian upper / lower |
| `bollinger_width` | Bollinger band width (+ upper/lower) |
| `ichimoku` | Tenkan, Kijun, Span A/B, cloud |
| `supertrend` | SuperTrend level + direction (±1) |
| `obv` | On-Balance Volume |
| `money_flow_index` | 14-period MFI |
| `chaikin_money_flow` | 20-period CMF |
| `accumulation_distribution` | A/D line |
| `volume_delta` | Up/down bar volume proxy |
| `cumulative_delta` | Cumulative volume delta |
| `tick_index` | Tape / single-name tick proxy |
| `advance_decline` | A/D breadth (or proxy) |
| `trin` | Arms Index (or proxy) |
| `breadth` | Breadth ratio ∈ [0,1] |
| `market_profile` | POC / VAH / VAL |
| `volume_profile` | Volume-at-price POC + bins |
| `liquidity_score` | Volume / range liquidity ∈ [0,1] |

Breadth fields accept an optional market-structure feed; otherwise single-name proxies are used.

## Macro Features (complete)

Module: `src/gamma_squeeze/features/macro_features.py`  
Endpoint: `POST|GET /v1/macro-features`  
Catalog: `MACRO_FEATURES_FIELDS`  
Source: Cloudflare KV **Fred-Economic-data** (+ CBOE delayed / OHLCV / structure.fx fallbacks)  
Worker: `https://fred-economic-data-icy-cloud-8332.2s6m8rz8fc.workers.dev`  
Local mirror: `EXPORT_FRED_ECONOMIC_DATA_SSD_PATH` or `/Volumes/PortableSSD/Fred Economic data`

| Feature | Source |
|---------|--------|
| `fed_funds` | `FEDFUNDS` |
| `cpi` | `CPIAUCSL` (+ `core_cpi` / YoY helpers) |
| `ppi` | `PPIACO` |
| `payrolls` | `PAYEMS` |
| `pmi` | Empire State / `NAPM` if synced |
| `gdp` | Real GDP growth `A191RL1Q225SBEA` |
| `consumer_sentiment` | `UMCSENT` |
| `treasury_curve` | `DGS2` / `DGS10` / `DGS30` (+ spreads) |
| `yield_spread` | `T10Y2Y` (else DGS10−DGS2) |
| `dollar_index` | `DTWEXBGS` (else structure.fx DXY) |
| `usdjpy` | `DEXJPUS` (else structure.fx) |
| `move` | KV `MOVE` or OHLCV `MOVE` |
| `vix` | `VIXCLS` / CBOE delayed |
| `vvix` | CBOE delayed |
| `economic_surprise_index` | calendar surprise → WEI Δ z-score → −STLFSI4 |
| `credit_spread` | `BAA10Y` |
| `corporate_bond_spread` | `BAMLC0A0CM` |
| `high_yield_spread` | `BAMLH0A0HYM2` |

Also emits `cpi_yoy`, `ppi_yoy`, `payrolls_mom` for ML.

## API

```bash
curl -X POST http://127.0.0.1:8002/v1/dealer-positioning \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL"}'

curl -X POST http://127.0.0.1:8002/v1/options-metrics \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL"}'

curl -X POST http://127.0.0.1:8002/v1/technical-indicators \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL"}'

curl -X POST http://127.0.0.1:8002/v1/macro-features \
  -H 'content-type: application/json' \
  -d '{"as_of":"2026-07-31"}'

curl 'http://127.0.0.1:8002/v1/features/AAPL?rebuild=true'
```

