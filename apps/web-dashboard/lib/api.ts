const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const DASH_BASE = process.env.NEXT_PUBLIC_DASH_BASE ?? "http://127.0.0.1:8011";

export async function runPipeline(symbol: string) {
  const url = `${API_BASE}/v1/pipeline/run?symbol=${encodeURIComponent(symbol)}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`pipeline failed: ${res.status}`);
  return res.json();
}

export async function stackHealth() {
  const res = await fetch(`${API_BASE}/v1/stack/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`stack health failed: ${res.status}`);
  return res.json();
}

/** Institutional dashboard JSON — Alpaca-forward panels + heatmaps */
export async function fetchDashboard(symbol: string, full = false) {
  const url = `${DASH_BASE}/v1/dashboard/${encodeURIComponent(symbol)}?use_live_alpaca=true&full=${full}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`dashboard failed: ${res.status}`);
  return res.json();
}

/** Dashboard meta via Alpaca API health */
export async function fetchDashboardMeta() {
  const res = await fetch(`${DASH_BASE}/v1/meta`, { cache: "no-store" });
  if (!res.ok) throw new Error(`dashboard meta failed: ${res.status}`);
  return res.json();
}

export function alertsWsUrl(symbol: string) {
  const base = process.env.NEXT_PUBLIC_WS_ALERTS ?? "ws://127.0.0.1:8012/v1/ws/alerts";
  const u = new URL(base);
  u.searchParams.set("symbol", symbol);
  return u.toString();
}
