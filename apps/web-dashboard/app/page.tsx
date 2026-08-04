"use client";

import { useEffect, useMemo, useState } from "react";
import { ProbabilityChart } from "@/components/ProbabilityChart";
import { HeatmapPanel } from "@/components/HeatmapPanel";
import {
  alertsWsUrl,
  fetchDashboard,
  fetchDashboardMeta,
  runPipeline,
  stackHealth,
} from "@/lib/api";

export default function HomePage() {
  const [symbol, setSymbol] = useState("AAPL");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dash, setDash] = useState<any>(null);
  const [meta, setMeta] = useState<any>(null);
  const [stack, setStack] = useState<any>(null);
  const [alertMsg, setAlertMsg] = useState<string>("waiting for alerts…");

  const panels = dash?.data?.panels ?? dash?.panels ?? {};
  const heatmaps = dash?.data?.heatmaps ?? dash?.heatmaps ?? {};
  const series = useMemo(() => panels?.probability?.term_structure ?? panels?.forecast?.term_structure ?? [], [panels]);

  async function onLoad() {
    setLoading(true);
    setError(null);
    try {
      const [d, m, s] = await Promise.all([
        fetchDashboard(symbol, false),
        fetchDashboardMeta().catch(() => null),
        stackHealth().catch(() => null),
      ]);
      setDash(d);
      setMeta(m);
      setStack(s);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onPipeline() {
    setLoading(true);
    setError(null);
    try {
      await runPipeline(symbol);
      await onLoad();
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setLoading(false);
    }
  }

  useEffect(() => {
    onLoad();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const ws = new WebSocket(alertsWsUrl(symbol));
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        setAlertMsg(
          data.fired
            ? (data.alerts?.[0]?.message ?? "alert fired")
            : `${symbol}: no alert (P below threshold)`,
        );
      } catch {
        setAlertMsg(String(ev.data));
      }
    };
    ws.onerror = () => setAlertMsg("WebSocket unavailable — start trade_alerts :8012");
    return () => ws.close();
  }, [symbol]);

  const regime = panels.market_regime || {};
  const forecast = panels.forecast || {};
  const gex = panels.gamma_exposure || panels.net_gex || {};
  const levels = panels.levels || {};
  const dealer = panels.dealer_positioning || {};
  const risk = panels.risk || {};
  const macro = panels.macro_dashboard || {};
  const alpaca = dash?.data?.alpaca ?? meta?.alpaca ?? {};

  return (
    <main>
      <h1>Institutional Desk</h1>
      <p className="muted">
        Regime · Dealer · GEX · Forecast · Surfaces — Alpaca-forward data plane
      </p>

      <div className="panel toolbar">
        <input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
        <button onClick={onLoad} disabled={loading}>
          {loading ? "Loading…" : "Load dashboard"}
        </button>
        <button onClick={onPipeline} disabled={loading}>
          Run pipeline
        </button>
        <span className="muted">{alertMsg}</span>
      </div>

      {error ? <div className="panel">Error: {error}</div> : null}

      <div className="kpis">
        <div className="kpi"><span>Regime</span><strong>{regime.current_state ?? "—"}</strong></div>
        <div className="kpi"><span>P(squeeze)</span><strong>{Number(forecast.probability ?? 0).toFixed(2)}</strong></div>
        <div className="kpi"><span>E[move]</span><strong>{Number(forecast.expected_move ?? 0).toFixed(2)}%</strong></div>
        <div className="kpi"><span>Net GEX</span><strong>{Number(gex.net_gex ?? gex.value ?? 0).toExponential(2)}</strong></div>
        <div className="kpi"><span>Flip</span><strong>{levels.gamma_flip?.level ?? "—"}</strong></div>
        <div className="kpi"><span>Call Wall</span><strong>{levels.call_wall?.level ?? "—"}</strong></div>
        <div className="kpi"><span>Put Wall</span><strong>{levels.put_wall?.level ?? "—"}</strong></div>
        <div className="kpi"><span>Alpaca</span><strong>{alpaca.ok ? "OK" : alpaca.configured ? "ERR" : "OFF"}</strong></div>
      </div>

      {forecast.why ? (
        <div className="panel">
          <h3>Why</h3>
          <p className="muted">{forecast.why}</p>
        </div>
      ) : null}

      <div className="row two">
        <div className="panel">
          <h3>Probability term structure</h3>
          <ProbabilityChart series={series} />
        </div>
        <div className="panel">
          <h3>Dealer / Risk / Macro</h3>
          <pre className="pre">
{JSON.stringify(
  {
    dealer: { delta: dealer.delta, exhaustion: dealer.exhaustion, liquidity: dealer.liquidity },
    risk: { iv: risk.atm_iv, rvol: risk.rvol_10d, stress: risk.positioning_stress },
    macro: { vix: macro.vix, fed_funds: macro.fed_funds, yield_spread: macro.yield_spread },
    patterns: panels.pattern_detection,
    calendar_events: panels.economic_calendar?.n_events,
    alerts: panels.alert_center,
  },
  null,
  2,
)}
          </pre>
        </div>
      </div>

      <div className="row two">
        <div className="panel">
          <HeatmapPanel
            title="Options Surface (OI)"
            x={heatmaps.options_surface?.strikes}
            y={heatmaps.options_surface?.dtes}
            z={heatmaps.options_surface?.oi}
          />
        </div>
        <div className="panel">
          <HeatmapPanel
            title="Volatility Surface"
            x={heatmaps.volatility_surface?.strikes}
            y={heatmaps.volatility_surface?.dtes}
            z={heatmaps.volatility_surface?.iv}
          />
        </div>
      </div>

      <div className="row two">
        <div className="panel">
          <HeatmapPanel
            title="Dealer Position Map"
            x={heatmaps.dealer_position_map?.strikes}
            y={heatmaps.dealer_position_map?.dtes}
            z={heatmaps.dealer_position_map?.dealer_gex}
          />
        </div>
        <div className="panel">
          <HeatmapPanel
            title="Liquidity Map"
            x={heatmaps.liquidity_map?.strikes}
            y={heatmaps.liquidity_map?.dtes}
            z={heatmaps.liquidity_map?.liquidity}
          />
        </div>
      </div>

      <div className="row two">
        <div className="panel">
          <HeatmapPanel
            title="3D Gamma (heatmap preview)"
            x={heatmaps.gamma_surface_3d?.strikes}
            y={heatmaps.gamma_surface_3d?.dtes}
            z={heatmaps.gamma_surface_3d?.gamma}
          />
        </div>
        <div className="panel">
          <HeatmapPanel
            title="Strike Distribution"
            kind="bar"
            x={heatmaps.strike_distribution?.strikes}
            y={heatmaps.strike_distribution?.total_oi}
          />
        </div>
      </div>

      <div className="row two">
        <div className="panel">
          <h3>Portfolio / Journal / Performance</h3>
          <pre className="pre">
{JSON.stringify(
  {
    portfolio: panels.portfolio,
    greeks: panels.greeks,
    journal: panels.trade_journal,
    performance: panels.performance_analytics,
  },
  null,
  2,
)}
          </pre>
        </div>
        <div className="panel">
          <h3>Alpaca health</h3>
          <pre className="pre">{JSON.stringify(alpaca || meta || stack, null, 2)}</pre>
        </div>
      </div>
    </main>
  );
}
