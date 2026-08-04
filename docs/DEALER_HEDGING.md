# Dealer Hedging Engine (Phase 9)

Service: `dealer_hedging` `:8007`  
Module: `src/gamma_squeeze/dealer/hedge_demand.py`

Simulates dealer Δ-hedging across a spot grid (dealers short customer-long options).

## Estimates

| Key | Label |
|-----|-------|
| `dealer_share_purchases` | Dealer Share Purchases |
| `dealer_share_sales` | Dealer Share Sales |
| `gamma_ramp` | Gamma Ramp |
| `strike_migration` | Strike Migration |
| `delta_hedging` | Delta Hedging |
| `dynamic_gamma` | Dynamic Gamma |
| `dealer_inventory` | Dealer Inventory |
| `dealer_liquidity` | Dealer Liquidity |
| `dealer_exhaustion` | Dealer Exhaustion |
| `dealer_position_flip` | Dealer Position Flip |

## Outputs

| Key | Label |
|-----|-------|
| `dealer_flow_map` | Dealer Flow Map |
| `dealer_demand_curve` | Dealer Demand Curve |
| `expected_hedging_volume` | Expected Hedging Volume |

## Method

1. Reprice contract greeks on a ±`grid_pct` spot lattice (`n_points`).
2. Hedge demand ≈ `−Σ delta_i × OI_i × 100` (dealer short convention).
3. Flow map = incremental share buys/sells between grid nodes.
4. Gamma ramp = `dGEX/dS` near spot; dynamic gamma = GEX path vs spot.
5. Strike migration tracks call/put walls at low / spot / high.
6. Exhaustion blends peak flow and \|hedge\| vs liquidity-scaled capacity.
7. Expected hedging volume = \|Δ hedge\| inside ± expected-move band.

Matrix origin: local SSD → Alpaca API → `alpaca-options-matrix-backup` KV.

## API

```bash
curl http://127.0.0.1:8007/v1/meta

curl -X POST http://127.0.0.1:8007/v1/simulate \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","grid_pct":0.08,"n_points":17}'

curl 'http://127.0.0.1:8007/v1/hedge-demand/AAPL?n_points=17'
```

## Phase 9 complete criteria

- All ten estimates populated from options matrix
- Flow map, demand curve, expected hedging volume returned
- Service API documents locked contract
- Tests pass; live symbol simulate succeeds
