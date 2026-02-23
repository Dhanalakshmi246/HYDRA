/**
 * api.js — Single source of truth for every backend URL
 *
 * The Vite dev-server proxy (vite.config.js) forwards each
 * prefix to the correct service port, so we only need relative paths.
 *
 * In production, the API Gateway (port 8000) would aggregate these.
 */

const API = Object.freeze({
  // ── Phase 1 ──────────────────────────────────
  predictions: '/api/v1/predictions/all',
  alertLog:    '/api/v1/alert/log',

  // ── Phase 2 ──────────────────────────────────
  causalRisk:  (villageId) => `/api/v1/causal/risk/${villageId}`,
  causalIntervene: '/api/v1/causal/intervene',

  chorusStats: '/api/v1/chorus/stats',
  chorusDemoGenerate: (villageId, count = 5) =>
    `/api/v1/chorus/demo/generate?village_id=${villageId}&count=${count}`,
  chorusWS: `ws://${typeof window !== 'undefined' ? window.location.hostname : 'localhost'}:8008/ws/signals`,

  evacPlan:    (scenarioId) => `/api/v1/evacuation/plan/${scenarioId}`,
  evacNotify:  '/api/v1/evacuation/notifications',
  evacDemo:    '/api/v1/evacuation/demo',
  evacCompute: '/api/v1/evacuation/compute',

  flStatus:    '/api/v1/fl/status',

  ledgerSummary: '/api/v1/ledger/chain/summary',
  ledgerChain:   '/api/v1/ledger/chain',
  ledgerVerify:  '/api/v1/ledger/verify',
  ledgerDemoFlood: (villageId) =>
    `/api/v1/ledger/demo/flood?village_id=${villageId}`,

  mirrorCF:     (eventId) => `/api/v1/mirror/event/${eventId}/counterfactuals`,
  mirrorCustom: (eventId) => `/api/v1/mirror/event/${eventId}/custom`,
  mirrorReport: (eventId) => `/api/v1/mirror/event/${eventId}/report`,

  // ── Phase 3 ──────────────────────────────────
  scarnetLatest:   '/api/v1/scarnet/latest',
  scarnetTrigger:  '/api/v1/scarnet/trigger-demo',
  scarnetHistory:  (id) => `/api/v1/scarnet/history/${id}`,
  scarnetTiles:    { before: '/api/v1/scarnet/tiles/before', after: '/api/v1/scarnet/tiles/after' },
  scarnetRiskDelta:(id) => `/api/v1/scarnet/risk-delta/${id}`,

  // API Gateway aggregated endpoints
  dashboardSnapshot: '/api/v1/dashboard/snapshot',
  gatewayHealth:     '/api/v1/dashboard/health',
})

export default API
