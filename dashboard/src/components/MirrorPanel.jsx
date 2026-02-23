/**
 * MirrorPanel — Counterfactual "What if?" scenario comparison panel.
 *
 * Shows preset scenarios and lets users run custom what-if queries against
 * the MIRROR service. Displays side-by-side base vs modified outcomes with
 * a timeline risk chart.
 */

import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const MIRROR_API = import.meta.env.VITE_MIRROR_API || '/api/v1/mirror'

// Demo timeline data for fallback
function generateDemoTimeline(factor = 1.0) {
  return Array.from({ length: 24 }, (_, i) => {
    const base = 0.15 + 0.4 * Math.sin((i / 24) * Math.PI) + (i > 12 ? 0.2 : 0)
    return {
      hour: i,
      base_risk: Math.min(1, Math.max(0, base)),
      modified_risk: Math.min(1, Math.max(0, base * factor)),
    }
  })
}

const PRESET_SCENARIOS = [
  { id: 'preset_half_rain', label: '50% less rain', icon: '🌤️' },
  { id: 'preset_double_rain', label: '2× rainfall', icon: '🌧️' },
  { id: 'preset_early_dam', label: 'Dam 4h early', icon: '🌊' },
  { id: 'preset_dry_soil', label: 'Dry soil', icon: '🏜️' },
  { id: 'preset_saturated_soil', label: 'Wet soil', icon: '💧' },
  { id: 'preset_no_dam', label: 'No dam release', icon: '🚫' },
]

export default function MirrorPanel({ demoMode = false }) {
  const [activePreset, setActivePreset] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState(true)
  const [chartData, setChartData] = useState([])

  const runScenario = useCallback(async (presetId) => {
    setActivePreset(presetId)
    setLoading(true)
    try {
      const { data } = await axios.post(`${MIRROR_API}/preset/${presetId}`)
      setResult(data)
      // Build chart data from timeline
      if (data.timeline && data.timeline.length > 0) {
        const chart = data.timeline.map((step, i) => ({
          hour: step.time_offset_hours || i,
          base_risk: data.base_outcome?.peak_risk
            ? (data.base_outcome.peak_risk * Math.sin((i / data.timeline.length) * Math.PI))
            : 0.3,
          modified_risk: step.risk_score || 0,
          water_level: step.water_level_m || 0,
        }))
        setChartData(chart)
      }
    } catch {
      // Demo fallback
      const factors = {
        preset_half_rain: 0.5,
        preset_double_rain: 2.0,
        preset_early_dam: 0.7,
        preset_dry_soil: 0.6,
        preset_saturated_soil: 1.4,
        preset_no_dam: 0.8,
      }
      const f = factors[presetId] || 1.0
      setChartData(generateDemoTimeline(f))
      setResult({
        risk_delta: f > 1 ? 0.15 : -0.12,
        lives_impact: f > 1 ? -250 : 350,
        base_outcome: { peak_risk: 0.72, peak_level_m: 5.2, danger_hours: 8 },
        modified_outcome: { peak_risk: 0.72 * f, peak_level_m: 5.2 * f, danger_hours: Math.round(8 * f) },
      })
    }
    setLoading(false)
  }, [])

  // Auto-run first preset in demo mode
  useEffect(() => {
    if (demoMode && !activePreset) {
      runScenario('preset_half_rain')
    }
  }, [demoMode, activePreset, runScenario])

  if (!expanded) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="bg-navy/90 border border-gray-700 text-purple-400 text-xs font-mono px-3 py-2 rounded-lg hover:border-purple-400 transition-colors"
      >
        MIRROR ▸
      </button>
    )
  }

  const riskDelta = result?.risk_delta || 0

  return (
    <div className="bg-navy/95 backdrop-blur-sm border border-gray-700 rounded-xl p-4 max-w-md">
      <div className="flex justify-between items-center mb-3">
        <h3 className="font-display text-sm text-white tracking-wider">
          MIRROR · <span className="text-purple-400">What If?</span>
        </h3>
        <button onClick={() => setExpanded(false)} className="text-gray-500 text-xs hover:text-white">✕</button>
      </div>

      {/* Preset scenario buttons */}
      <div className="grid grid-cols-3 gap-1.5 mb-3">
        {PRESET_SCENARIOS.map((s) => (
          <button
            key={s.id}
            onClick={() => runScenario(s.id)}
            className={`text-[10px] px-2 py-1.5 rounded border transition-colors ${
              activePreset === s.id
                ? 'bg-purple-900/50 border-purple-500 text-purple-300'
                : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-500'
            }`}
          >
            {s.icon} {s.label}
          </button>
        ))}
      </div>

      {/* Loading */}
      {loading && (
        <div className="text-center py-4 text-gray-500 text-xs">
          Running counterfactual simulation...
        </div>
      )}

      {/* Timeline chart */}
      {chartData.length > 0 && !loading && (
        <div className="mb-3">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Risk Timeline</div>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis
                dataKey="hour"
                tick={{ fontSize: 9, fill: '#666' }}
                label={{ value: 'Hours', position: 'insideBottomRight', fontSize: 9, fill: '#555' }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fontSize: 9, fill: '#666' }}
                width={30}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333', fontSize: 10 }}
                labelFormatter={(v) => `T+${v}h`}
              />
              <Line
                type="monotone"
                dataKey="base_risk"
                stroke="#f97316"
                strokeWidth={1.5}
                strokeDasharray="4 4"
                dot={false}
                name="Base"
              />
              <Line
                type="monotone"
                dataKey="modified_risk"
                stroke="#a855f7"
                strokeWidth={2}
                dot={false}
                name="Modified"
              />
              <Legend wrapperStyle={{ fontSize: 9 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Outcome comparison */}
      {result && !loading && (
        <div className="grid grid-cols-2 gap-2 mb-2">
          <div className="bg-gray-800/50 rounded p-2">
            <div className="text-[10px] text-orange-400 uppercase mb-1">Base</div>
            <div className="text-xs text-white font-mono">
              {((result.base_outcome?.peak_risk || 0) * 100).toFixed(1)}% risk
            </div>
            <div className="text-[10px] text-gray-500">
              {result.base_outcome?.peak_level_m?.toFixed(1) || '—'}m peak ·{' '}
              {result.base_outcome?.danger_hours || 0}h danger
            </div>
          </div>
          <div className="bg-gray-800/50 rounded p-2">
            <div className="text-[10px] text-purple-400 uppercase mb-1">Modified</div>
            <div className="text-xs text-white font-mono">
              {((result.modified_outcome?.peak_risk || 0) * 100).toFixed(1)}% risk
            </div>
            <div className="text-[10px] text-gray-500">
              {result.modified_outcome?.peak_level_m?.toFixed(1) || '—'}m peak ·{' '}
              {result.modified_outcome?.danger_hours || 0}h danger
            </div>
          </div>
        </div>
      )}

      {/* Delta summary */}
      {result && !loading && (
        <div className={`text-center text-xs font-mono py-1 rounded ${
          riskDelta > 0 ? 'bg-red-900/30 text-red-400' : 'bg-green-900/30 text-green-400'
        }`}>
          Risk Δ: {riskDelta > 0 ? '+' : ''}{(riskDelta * 100).toFixed(1)}%
          {result.lives_impact != null && (
            <span className="ml-2 text-gray-400">
              · {result.lives_impact > 0 ? '+' : ''}{result.lives_impact} lives
            </span>
          )}
        </div>
      )}
    </div>
  )
}
