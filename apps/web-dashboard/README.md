# Visualization Dashboard (Next.js)

Phase 2 locked visualization stack:

| Library | Package |
|---------|---------|
| **React** | `react`, `react-dom` |
| **Next.js** | `next` |
| **Plotly** | `plotly.js`, `react-plotly.js` |
| **TradingView Lightweight Charts** | `lightweight-charts` |
| **ECharts** | `echarts`, `echarts-for-react` |

Backend JSON panels also come from the FastAPI `dashboard` service (`:8011`). Live alerts use **WebSockets** on the Trade Alert API (`:8012`).

## Run

```bash
npm install
npm run dev
```

Point the UI at the orchestrator (`http://127.0.0.1:8000`) via env as documented in the app config.
