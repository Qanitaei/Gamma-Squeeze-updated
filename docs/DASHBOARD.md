# Visualization Dashboard

Service: `dashboard` `:8011`  
Module: `src/gamma_squeeze/viz/institutional_dashboard.py`  
Web: `apps/web-dashboard` (Next.js)

Data plane: **Alpaca API** for options surfaces / health (SSD fallback).  
**No HTML desk. No Schwab.**

## Panels

Market Regime · Dealer Positioning · Gamma Exposure · Net GEX · Gamma Flip · Call Wall · Put Wall · Forecast · Probability · Expected Move · Pattern Detection · Portfolio · Greeks · Risk · Macro Dashboard · Economic Calendar · Alert Center · Trade Journal · Performance Analytics

## Interactive heatmaps

Options Surface · 3D Gamma Surface · Volatility Surface · Dealer Position Map · Liquidity Map · Strike Distribution

## Usage

```python
from services.dashboard.service import build_dashboard, meta

meta()  # Alpaca health + catalog
dash = build_dashboard("AAPL", use_live_alpaca=True, full=False)
# dash["panels"], dash["heatmaps"], dash["plotly"]
```

JSON: `GET /v1/dashboard/AAPL`  
Set `full=true` to enrich with regime / patterns / hedges / alerts / dealer simulation.
