/**
 * EvacuationMap — Evacuation route overlay with zone markers and
 * animated evacuation flow arrows.
 *
 * Fetches from /api/v1/evacuation/graph/{village} and
 * /api/v1/evacuation/plan when risk is high.
 */

import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const EVAC_API = import.meta.env.VITE_EVAC_API || '/api/v1/evacuation'

const ZONE_COLORS = {
  safe: '#22c55e',       // green
  populated: '#f97316',  // orange
  at_risk: '#ef4444',    // red
}

const ROUTE_COLORS = {
  paved: '#60a5fa',
  unpaved: '#fbbf24',
  bridge: '#a78bfa',
  boat: '#34d399',
}

export default function EvacuationMap({ selectedVillage = 'kullu_01', riskScore = 0, demoMode = false }) {
  const [graph, setGraph] = useState(null)
  const [plan, setPlan] = useState(null)
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState(true)

  // Fetch graph
  const fetchGraph = useCallback(async () => {
    try {
      const { data } = await axios.get(`${EVAC_API}/graph/${selectedVillage}`)
      setGraph(data)
    } catch {
      // Demo fallback
      setGraph({
        village_id: selectedVillage,
        zones: [
          { zone_id: 'z1', name: 'Lower Colony', population: 320, is_safe_zone: false, elevation_m: 1180 },
          { zone_id: 'z2', name: 'Market Area', population: 450, is_safe_zone: false, elevation_m: 1195 },
          { zone_id: 'z3', name: 'River Bank', population: 180, is_safe_zone: false, elevation_m: 1170 },
          { zone_id: 'z5', name: 'School Hill', population: 0, is_safe_zone: true, capacity: 800, elevation_m: 1240 },
          { zone_id: 'z6', name: 'Temple Grounds', population: 0, is_safe_zone: true, capacity: 600, elevation_m: 1260 },
        ],
        routes: [
          { route_id: 'r1', from_zone: 'z1', to_zone: 'z5', distance_km: 1.2, travel_time_min: 15, road_type: 'paved', is_passable: true },
          { route_id: 'r5', from_zone: 'z3', to_zone: 'z6', distance_km: 0.6, travel_time_min: 10, road_type: 'unpaved', is_passable: true },
          { route_id: 'r3', from_zone: 'z2', to_zone: 'z5', distance_km: 0.9, travel_time_min: 12, road_type: 'paved', is_passable: true },
        ],
        total_population: 950,
        safe_capacity: 1400,
      })
    }
  }, [selectedVillage])

  // Fetch plan when risk is high
  const fetchPlan = useCallback(async () => {
    if (riskScore < 0.4 && !demoMode) {
      setPlan(null)
      return
    }
    try {
      const { data } = await axios.post(`${EVAC_API}/plan`, null, {
        params: { village_id: selectedVillage, risk_score: demoMode ? 0.75 : riskScore },
      })
      setPlan(data)
    } catch {
      setPlan(null)
    }
  }, [selectedVillage, riskScore, demoMode])

  useEffect(() => { fetchGraph() }, [fetchGraph])
  useEffect(() => { fetchPlan() }, [fetchPlan])

  if (!expanded) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="bg-navy/90 border border-gray-700 text-accent text-xs font-mono px-3 py-2 rounded-lg hover:border-accent transition-colors"
      >
        EVAC MAP ▸
      </button>
    )
  }

  return (
    <div className="bg-navy/95 backdrop-blur-sm border border-gray-700 rounded-xl p-4 max-w-sm">
      <div className="flex justify-between items-center mb-3">
        <h3 className="font-display text-sm text-white tracking-wider">
          EVACUATION MAP
        </h3>
        <button onClick={() => setExpanded(false)} className="text-gray-500 text-xs hover:text-white">✕</button>
      </div>

      {/* Zone list */}
      <div className="space-y-2 mb-3">
        <div className="text-[10px] text-gray-500 uppercase tracking-wider">Zones</div>
        {graph?.zones?.map((z) => (
          <div
            key={z.zone_id}
            className="flex items-center justify-between bg-gray-800/50 rounded px-2 py-1"
          >
            <div className="flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: z.is_safe_zone ? ZONE_COLORS.safe : ZONE_COLORS.populated }}
              />
              <span className="text-xs text-gray-300">{z.name}</span>
            </div>
            <span className="text-[10px] text-gray-500 font-mono">
              {z.is_safe_zone
                ? `CAP ${z.capacity}`
                : `POP ${z.population}`
              }
              {' '} · {z.elevation_m}m
            </span>
          </div>
        ))}
      </div>

      {/* Routes */}
      {graph?.routes?.length > 0 && (
        <div className="space-y-1 mb-3">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider">Routes</div>
          {graph.routes.map((r) => (
            <div
              key={r.route_id}
              className="flex items-center gap-2 text-[10px]"
            >
              <div
                className="w-8 h-0.5"
                style={{ backgroundColor: ROUTE_COLORS[r.road_type] || '#666' }}
              />
              <span className="text-gray-400">
                {r.from_zone} → {r.to_zone}
              </span>
              <span className="text-gray-600">
                {r.distance_km}km · {r.travel_time_min}min · {r.road_type}
              </span>
              {!r.is_passable && <span className="text-red-500 font-bold">BLOCKED</span>}
            </div>
          ))}
        </div>
      )}

      {/* Plan */}
      {plan && plan.actions?.length > 0 && (
        <div className="border-t border-gray-700 pt-2 mt-2">
          <div className="text-[10px] text-amber-500 uppercase tracking-wider mb-1">
            RL Evacuation Plan · Reward: {plan.rl_reward}
          </div>
          {plan.actions.slice(0, 5).map((a, i) => (
            <div key={a.action_id} className="text-[10px] text-gray-300 flex gap-1 mb-0.5">
              <span className="text-accent font-mono">P{a.priority}</span>
              <span>{a.zone_id} → {a.recommended_route}</span>
              <span className="text-gray-500">{a.population_to_move} people · {a.estimated_travel_min}min</span>
            </div>
          ))}
          <div className="text-[10px] text-gray-500 mt-1">
            Clear time: ~{plan.estimated_clear_time_min}min · {plan.actions.length} moves
          </div>
        </div>
      )}

      {/* Summary */}
      <div className="flex gap-3 mt-3 text-[10px]">
        <span className="text-green-400">■ SAFE</span>
        <span className="text-orange-400">■ POPULATED</span>
        <span className="text-gray-500">
          Pop: {graph?.total_population || 0} · Cap: {graph?.safe_capacity || 0}
        </span>
      </div>
    </div>
  )
}
