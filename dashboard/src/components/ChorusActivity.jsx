/**
 * ChorusActivity — Community intelligence feed showing live
 * community sentiment, report counts, and CHORUS aggregation.
 */

import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const CHORUS_API = import.meta.env.VITE_CHORUS_API || '/api/v1/chorus'

const SENTIMENT_BADGES = {
  CALM:      { bg: 'bg-green-900/40', border: 'border-green-700', text: 'text-green-400', icon: '😊' },
  CONCERNED: { bg: 'bg-yellow-900/40', border: 'border-yellow-700', text: 'text-yellow-400', icon: '😟' },
  ANXIOUS:   { bg: 'bg-orange-900/40', border: 'border-orange-700', text: 'text-orange-400', icon: '😰' },
  PANIC:     { bg: 'bg-red-900/40', border: 'border-red-700', text: 'text-red-400', icon: '🚨' },
}

export default function ChorusActivity({ selectedVillage = null, demoMode = false }) {
  const [stats, setStats] = useState(null)
  const [expanded, setExpanded] = useState(true)
  const [generating, setGenerating] = useState(false)

  const fetchStats = useCallback(async () => {
    try {
      const { data } = await axios.get(`${CHORUS_API}/stats`)
      setStats(data)
    } catch {
      // Demo fallback
      setStats({
        total_reports: 42,
        villages_reporting: 5,
        aggregations: {
          kullu_01: {
            village_id: 'kullu_01', report_count: 12, dominant_sentiment: 'ANXIOUS',
            panic_ratio: 0.167, avg_credibility: 0.62, flood_mention_rate: 0.75,
            community_risk_boost: 0.108, top_keywords: ['baadh', 'paani', 'bachao', 'khatra'],
          },
          mandi_01: {
            village_id: 'mandi_01', report_count: 8, dominant_sentiment: 'CONCERNED',
            panic_ratio: 0.0, avg_credibility: 0.55, flood_mention_rate: 0.5,
            community_risk_boost: 0.05, top_keywords: ['baarish', 'nadi'],
          },
          majuli_01: {
            village_id: 'majuli_01', report_count: 15, dominant_sentiment: 'PANIC',
            panic_ratio: 0.4, avg_credibility: 0.71, flood_mention_rate: 0.93,
            community_risk_boost: 0.173, top_keywords: ['flood', 'submerged', 'rescue', 'emergency', 'trapped'],
          },
          dhemaji_01: {
            village_id: 'dhemaji_01', report_count: 4, dominant_sentiment: 'CALM',
            panic_ratio: 0.0, avg_credibility: 0.45, flood_mention_rate: 0.25,
            community_risk_boost: 0.025, top_keywords: ['normal'],
          },
          sujanpur_01: {
            village_id: 'sujanpur_01', report_count: 3, dominant_sentiment: 'CONCERNED',
            panic_ratio: 0.0, avg_credibility: 0.5, flood_mention_rate: 0.33,
            community_risk_boost: 0.033, top_keywords: ['baarish'],
          },
        },
      })
    }
  }, [])

  const generateDemo = useCallback(async () => {
    setGenerating(true)
    try {
      await axios.post(`${CHORUS_API}/demo/generate`, null, {
        params: { village_id: selectedVillage || 'kullu_01', count: 5 },
      })
    } catch { /* ignore */ }
    await fetchStats()
    setGenerating(false)
  }, [selectedVillage, fetchStats])

  useEffect(() => { fetchStats() }, [fetchStats])
  useEffect(() => {
    const iv = setInterval(fetchStats, demoMode ? 8000 : 30000)
    return () => clearInterval(iv)
  }, [fetchStats, demoMode])

  if (!expanded) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="bg-navy/90 border border-gray-700 text-emerald-400 text-xs font-mono px-3 py-2 rounded-lg hover:border-emerald-400 transition-colors"
      >
        CHORUS ▸
      </button>
    )
  }

  const aggregations = stats?.aggregations ? Object.values(stats.aggregations) : []
  // Sort by panic_ratio descending
  aggregations.sort((a, b) => (b.panic_ratio || 0) - (a.panic_ratio || 0))

  return (
    <div className="bg-navy/95 backdrop-blur-sm border border-gray-700 rounded-xl p-4 max-w-xs">
      <div className="flex justify-between items-center mb-3">
        <h3 className="font-display text-sm text-white tracking-wider">
          <span className="text-emerald-400">CHORUS</span> · Community Intel
        </h3>
        <button onClick={() => setExpanded(false)} className="text-gray-500 text-xs hover:text-white">✕</button>
      </div>

      {/* Global stats */}
      <div className="flex gap-4 mb-3">
        <div>
          <div className="text-lg text-white font-mono">{stats?.total_reports || 0}</div>
          <div className="text-[9px] text-gray-500 uppercase">Reports</div>
        </div>
        <div>
          <div className="text-lg text-white font-mono">{stats?.villages_reporting || 0}</div>
          <div className="text-[9px] text-gray-500 uppercase">Villages</div>
        </div>
        {demoMode && (
          <button
            onClick={generateDemo}
            disabled={generating}
            className="ml-auto text-[10px] text-emerald-500 hover:text-emerald-300 border border-emerald-700 rounded px-2 py-1 disabled:opacity-50"
          >
            {generating ? '...' : '+ Demo'}
          </button>
        )}
      </div>

      {/* Village sentiment cards */}
      <div className="space-y-1.5 max-h-60 overflow-y-auto">
        {aggregations.map((agg) => {
          const badge = SENTIMENT_BADGES[agg.dominant_sentiment] || SENTIMENT_BADGES.CALM
          return (
            <div
              key={agg.village_id}
              className={`${badge.bg} border ${badge.border} rounded px-2 py-1.5 ${
                selectedVillage === agg.village_id ? 'ring-1 ring-white/30' : ''
              }`}
            >
              <div className="flex justify-between items-center">
                <span className="text-xs text-white">{agg.village_id}</span>
                <span className={`text-[10px] font-mono ${badge.text}`}>
                  {badge.icon} {agg.dominant_sentiment}
                </span>
              </div>
              <div className="flex gap-2 mt-0.5 text-[9px] text-gray-400">
                <span>{agg.report_count} reports</span>
                <span>·</span>
                <span>flood: {(agg.flood_mention_rate * 100).toFixed(0)}%</span>
                <span>·</span>
                <span>panic: {(agg.panic_ratio * 100).toFixed(0)}%</span>
              </div>
              {agg.community_risk_boost > 0.05 && (
                <div className="text-[9px] text-amber-400 mt-0.5">
                  ⚠ Risk boost: +{(agg.community_risk_boost * 100).toFixed(1)}%
                </div>
              )}
              {agg.top_keywords?.length > 0 && (
                <div className="flex gap-1 mt-1 flex-wrap">
                  {agg.top_keywords.slice(0, 4).map((kw) => (
                    <span key={kw} className="text-[8px] bg-gray-800 rounded px-1 py-0.5 text-gray-500">
                      {kw}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )
        })}
        {aggregations.length === 0 && (
          <div className="text-xs text-gray-600 text-center py-3">
            No community reports yet
          </div>
        )}
      </div>
    </div>
  )
}
