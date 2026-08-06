/**
 * Gamma Squeeze Data API — Cloudflare Worker
 *
 * Binding pattern (same idea as D1 `env.MY_DB`):
 *   env.KV  →  Cloudflare KV namespace `gamma-squeeze-data`
 *
 * Worker: https://gamma-squeeze-data-icy-shadow-db40.2s6m8rz8fc.workers.dev/
 * OpenAPI 3.0.3 at `/` and `/openapi.json`.
 */

import OPENAPI from "../openapi.json";

/** @typedef {{ success?: boolean, symbol?: string, squeeze_probability?: number, [k: string]: any }} SqueezeRow */

/**
 * @typedef {Object} Env
 * @property {KVNamespace} KV  Bound to KV namespace `gamma-squeeze-data`
 * @property {string} [WORKER_URL]
 * @property {string} [NAMESPACE_NAME]
 */

const NAMESPACE_NAME = "gamma-squeeze-data";
const WORKER_URL = "https://gamma-squeeze-data-icy-shadow-db40.2s6m8rz8fc.workers.dev";
const TICKER_PATTERN = "[A-Z.]{1,10}";
/** Date-only forecast keys: AAPL/forecast/2026-08-06 (no T061106Z suffixes) */
const DATE_PATTERN = "\\d{4}-\\d{2}-\\d{2}";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

/**
 * @param {unknown} data
 * @param {number} [status]
 */
function json(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      ...corsHeaders,
      "content-type": "application/json; charset=utf-8",
      "cache-control": "public, max-age=60",
    },
  });
}

/** @param {string} msg */
function notFound(msg) {
  return json({ success: false, error: msg }, 404);
}

/**
 * Read JSON from the bound KV namespace (env.KV → gamma-squeeze-data).
 * Mirrors the D1 pattern: await env.MY_DB.prepare(...).run()
 *                          await env.KV.get(key)
 *
 * @param {Env} env
 * @param {string} key
 * @returns {Promise<any | null>}
 */
async function kvGet(env, key) {
  if (!env.KV) return null;
  const raw = await env.KV.get(key);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return { raw };
  }
}

/**
 * List up to 100 keys from the bound KV (KV analogue of SELECT … LIMIT 100).
 * @param {Env} env
 * @param {string} [prefix]
 */
async function kvListRows(env, prefix = "") {
  if (!env.KV) {
    return { success: false, error: "KV binding missing", results: [] };
  }
  const listed = await env.KV.list({ prefix, limit: 100 });
  /** @type {Array<{ key: string, expiration?: number, metadata?: any }>} */
  const results = (listed.keys || []).map((k) => ({
    key: k.name,
    expiration: k.expiration,
    metadata: k.metadata,
  }));
  return {
    success: true,
    binding: "KV",
    namespace: NAMESPACE_NAME,
    prefix,
    list_complete: listed.list_complete,
    cursor: listed.cursor,
    results,
  };
}

/** @param {string} baseUrl */
function buildRetrievals(baseUrl) {
  const items = [];
  for (const [path, methods] of Object.entries(OPENAPI.paths || {})) {
    for (const [method, spec] of Object.entries(methods)) {
      items.push({
        method: method.toUpperCase(),
        path,
        operationId: spec.operationId,
        summary: spec.summary,
        href: `${baseUrl}${path}`,
      });
    }
  }
  return items;
}

/**
 * @param {Env} env
 * @param {string} baseUrl
 */
function buildHealth(env, baseUrl) {
  return {
    success: true,
    openapi: "3.0.3",
    service: "gamma-squeeze-data",
    binding: "KV",
    namespace: NAMESPACE_NAME,
    kv_connected: Boolean(env.KV),
    worker: WORKER_URL,
    openapi_url: `${baseUrl}/openapi.json`,
    base44_import_url: `${baseUrl}/openapi.json`,
    retrievals_url: `${baseUrl}/retrievals`,
    env_pattern: "env.KV  // Cloudflare KV namespace gamma-squeeze-data",
    routes: [
      "/health",
      "/openapi.json",
      "/retrievals",
      "/v1/index",
      "/v1/keys",
      "/v1/scans/phase14/latest",
      "/v1/scans/phase14/findings",
      "/v1/scans/phase14/{YYYY-MM-DD}",
      "/v1/scans/phase14/by-date/{YYYY-MM-DD}/runs",
      "/v1/{symbol}/pipeline/latest",
      "/v1/{symbol}/pipeline/summary",
      "/v1/{symbol}/pipeline/{YYYY-MM-DD}",
      "/v1/{symbol}/forecast/latest",
      "/v1/{symbol}/forecast/{YYYY-MM-DD}",
      "/v1/extractions/{YYYY-MM-DD}",
      "/v1/extractions/{YYYY-MM-DD}/runs",
      "/v1/forecasts/index",
    ],
    kv_key_format: {
      index: "index",
      scan_latest: "scans/phase14_pipeline/latest",
      scan_findings: "scans/phase14_pipeline/findings",
      scan_dated: "scans/phase14_pipeline/{YYYY-MM-DD}",
      scan_day_runs: "scans/phase14_pipeline/by-date/{YYYY-MM-DD}/runs",
      symbol_pipeline: "{TICKER}/pipeline/latest",
      symbol_pipeline_summary: "{TICKER}/pipeline/summary",
      symbol_pipeline_dated: "{TICKER}/pipeline/{YYYY-MM-DD}",
      forecast_latest: "{TICKER}/latest",
      forecast_dated: "{TICKER}/forecast/{YYYY-MM-DD}",
      extraction_symbol: "extractions/{YYYY-MM-DD}/{TICKER}",
      extraction_index: "extractions/{YYYY-MM-DD}/index",
      extraction_day_runs: "extractions/{YYYY-MM-DD}/runs",
      forecasts_catalog: "forecasts/index",
    },
  };
}

export default {
  /**
   * @param {Request} request
   * @param {Env} env
   */
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }
    if (request.method !== "GET") {
      return json({ success: false, error: "Method not allowed" }, 405);
    }

    const url = new URL(request.url);
    const baseUrl = `${url.protocol}//${url.host}`;
    const path = url.pathname.replace(/\/+$/, "") || "/";

    // OpenAPI 3.0 at root
    if (path === "/" || path === "/openapi.json") {
      return json({
        ...OPENAPI,
        servers: [{ url: baseUrl, description: "Production Workers KV" }],
      });
    }

    if (path === "/health") {
      return json(buildHealth(env, baseUrl));
    }

    if (path === "/retrievals") {
      const items = buildRetrievals(baseUrl);
      return json({
        success: true,
        count: items.length,
        openapi: "3.0.3",
        base44: { openapi_import_url: `${baseUrl}/openapi.json` },
        items,
      });
    }

    // KV list — same shape idea as:
    //   await env.MY_DB.prepare("SELECT … LIMIT 100").run()
    //   await env.KV.list({ limit: 100 })
    if (path === "/v1/keys") {
      const prefix = url.searchParams.get("prefix") || "";
      return json(await kvListRows(env, prefix));
    }

    if (path === "/v1" || path === "/v1/index") {
      const index = (await kvGet(env, "index")) || {
        success: true,
        service: "gamma-squeeze-data",
        namespace: NAMESPACE_NAME,
        binding: "KV",
        openapi: "3.0.3",
        endpoints: buildHealth(env, baseUrl).routes,
      };
      return json(index);
    }

    if (path === "/v1/scans/phase14/latest") {
      const payload = await kvGet(env, "scans/phase14_pipeline/latest");
      if (!payload) return notFound("No phase14 pipeline scan uploaded");
      return json(payload);
    }

    if (path === "/v1/scans/phase14/findings") {
      const payload = await kvGet(env, "scans/phase14_pipeline/findings");
      if (!payload) return notFound("No phase14 findings uploaded");
      return json(payload);
    }

    if (path === "/v1/forecasts/index") {
      const payload = await kvGet(env, "forecasts/index");
      if (!payload) return notFound("No forecasts catalog uploaded");
      return json(payload);
    }

    const scanDayRunsRe = new RegExp(
      `^/v1/scans/phase14/by-date/(${DATE_PATTERN})/runs$`,
    );
    const scanHistoricalRe = new RegExp(
      `^/v1/scans/phase14/(${DATE_PATTERN})$`,
    );
    const dayRunsRe = new RegExp(`^/v1/extractions/(${DATE_PATTERN})/runs$`);
    const extractionIndexRe = new RegExp(`^/v1/extractions/(${DATE_PATTERN})$`);
    const pipelineLatestRe = new RegExp(
      `^/v1/(${TICKER_PATTERN})/pipeline/latest$`,
    );
    const pipelineSummaryRe = new RegExp(
      `^/v1/(${TICKER_PATTERN})/pipeline/summary$`,
    );
    // Date-only: /v1/AAPL/pipeline/2026-08-06 (no T153655Z)
    const pipelineHistoricalRe = new RegExp(
      `^/v1/(${TICKER_PATTERN})/pipeline/(${DATE_PATTERN})$`,
    );
    const latestRe = new RegExp(`^/v1/(${TICKER_PATTERN})/forecast/latest$`);
    const datedRe = new RegExp(
      `^/v1/(${TICKER_PATTERN})/forecast/(${DATE_PATTERN})$`,
    );

    let m = path.match(scanDayRunsRe);
    if (m) {
      const payload = await kvGet(
        env,
        `scans/phase14_pipeline/by-date/${m[1]}/runs`,
      );
      if (!payload) return notFound(`No phase14 runs for ${m[1]}`);
      return json(payload);
    }

    m = path.match(scanHistoricalRe);
    if (m) {
      const payload = await kvGet(env, `scans/phase14_pipeline/${m[1]}`);
      if (!payload) return notFound(`No phase14 scan ${m[1]}`);
      return json(payload);
    }

    m = path.match(dayRunsRe);
    if (m) {
      const payload = await kvGet(env, `extractions/${m[1]}/runs`);
      if (!payload) return notFound(`No extraction runs for ${m[1]}`);
      return json(payload);
    }

    m = path.match(extractionIndexRe);
    if (m) {
      const payload = await kvGet(env, `extractions/${m[1]}/index`);
      if (!payload) return notFound(`No extraction index for ${m[1]}`);
      return json(payload);
    }

    m = path.match(pipelineLatestRe);
    if (m) {
      const symbol = m[1].toUpperCase();
      /** @type {SqueezeRow | null} */
      const payload = await kvGet(env, `${symbol}/pipeline/latest`);
      if (!payload) return notFound(`No pipeline findings for ${symbol}`);
      return json(payload);
    }

    m = path.match(pipelineSummaryRe);
    if (m) {
      const symbol = m[1].toUpperCase();
      const payload = await kvGet(env, `${symbol}/pipeline/summary`);
      if (!payload) return notFound(`No pipeline summary for ${symbol}`);
      return json(payload);
    }

    m = path.match(pipelineHistoricalRe);
    if (m) {
      const symbol = m[1].toUpperCase();
      const date = m[2];
      const payload = await kvGet(env, `${symbol}/pipeline/${date}`);
      if (!payload) return notFound(`No pipeline for ${symbol} on ${date}`);
      return json(payload);
    }

    m = path.match(latestRe);
    if (m) {
      const symbol = m[1].toUpperCase();
      /** @type {SqueezeRow | null} */
      const payload = await kvGet(env, `${symbol}/latest`);
      if (!payload) return notFound(`No latest forecast for ${symbol}`);
      return json(payload);
    }

    m = path.match(datedRe);
    if (m) {
      const symbol = m[1].toUpperCase();
      const date = m[2];
      const payload = await kvGet(env, `${symbol}/forecast/${date}`);
      if (!payload) return notFound(`No forecast for ${symbol} on ${date}`);
      return json(payload);
    }

    return notFound(`Unknown path: ${path}`);
  },
};
