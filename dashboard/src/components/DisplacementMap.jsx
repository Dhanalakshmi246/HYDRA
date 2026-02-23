import { useState, useEffect } from 'react'

/**
 * DisplacementMap — Shelter occupancy + displacement flow tracker
 *
 * Gap 8 closure: Post-disaster displacement tracking with shelter
 * capacities, origin→shelter flows, and relief distribution status.
 *
 * Wired to:
 *   GET /api/v1/displacement/summary    (DisplacementTracker)
 *   GET /api/v1/displacement/shelters   (DisplacementTracker)
 *   GET /api/v1/displacement/flows      (DisplacementTracker)
 */

const DisplacementMap = () => {
  const [summary, setSummary] = useState(null)
  const [shelters, setShelters] = useState([])
  const [flows, setFlows] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [sRes, shRes, fRes] = await Promise.allSettled([
          fetch('/api/v1/displacement/summary'),
          fetch('/api/v1/displacement/shelters'),
          fetch('/api/v1/displacement/flows'),
        ])
        if (sRes.status === 'fulfilled' && sRes.value.ok) {
          setSummary(await sRes.value.json())
        }
        if (shRes.status === 'fulfilled' && shRes.value.ok) {
          const shData = await shRes.value.json()
          setShelters(shData.shelters || [])
        }
        if (fRes.status === 'fulfilled' && fRes.value.ok) {
          const fData = await fRes.value.json()
          setFlows(fData.flows || [])
        }
      } catch { /* ignore */ }
      setLoading(false)
    }
    fetchAll()
    const interval = setInterval(fetchAll, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="bg-slate-900 rounded-2xl p-5 border border-cyan-900/50">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-cyan-400 font-heading font-bold text-xl">
            Displacement Tracker
          </h2>
          <p className="text-slate-500 text-xs mt-0.5">
            Post-event shelter occupancy, displacement flows &amp; relief distribution
          </p>
        </div>
        {summary && (
          <span className="px-3 py-1 rounded-full text-xs font-bold bg-orange-900/40 text-orange-300 border border-orange-600/30">
            {summary.total_displaced?.toLocaleString()} displaced
          </span>
        )}
      </div>

      {loading ? (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-12 bg-slate-800 rounded-lg animate-pulse" />
          ))}
        </div>
      ) : (
        <>
          {/* Summary grid */}
          {summary && (
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 mb-5">
              {[
                { label: 'Displaced', value: summary.total_displaced?.toLocaleString(), color: 'text-orange-300' },
                { label: 'Sheltered', value: summary.currently_sheltered?.toLocaleString(), color: 'text-green-300' },
                { label: 'Children', value: summary.children_displaced, color: 'text-yellow-300' },
                { label: 'Elderly', value: summary.elderly_displaced, color: 'text-yellow-300' },
                { label: 'Medical', value: summary.medical_needs, color: 'text-red-300' },
                { label: 'Relief Runs', value: summary.relief_distributions, color: 'text-cyan-300' },
              ].map((item, i) => (
                <div key={i} className="bg-slate-800/60 rounded-xl p-3 text-center">
                  <p className={`text-xl font-bold font-heading ${item.color}`}>
                    {item.value ?? '—'}
                  </p>
                  <p className="text-slate-500 text-xs mt-0.5">{item.label}</p>
                </div>
              ))}
            </div>
          )}

          {/* Shelter capacity */}
          <div className="mb-5">
            <h3 className="text-slate-400 text-xs font-bold uppercase tracking-wider mb-2">
              Shelter Status
            </h3>
            <div className="space-y-2">
              {shelters.map((s, i) => {
                const pct = Math.round((s.current_occupancy / Math.max(1, s.capacity)) * 100)
                return (
                  <div key={i} className="bg-slate-800/60 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-slate-200 text-sm font-bold">{s.name}</span>
                      <span className="text-slate-500 text-xs">{s.district}</span>
                    </div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="text-slate-400 text-xs">
                        {s.current_occupancy} / {s.capacity}
                      </span>
                      <span className={`text-xs font-bold ${
                        pct > 90 ? 'text-red-400' : pct > 70 ? 'text-amber-400' : 'text-green-400'
                      }`}>
                        {pct}%
                      </span>
                      <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                        s.status === 'FULL' ? 'bg-red-900/40 text-red-300' :
                        s.status === 'OPEN' ? 'bg-green-900/40 text-green-300' :
                        'bg-slate-700 text-slate-400'
                      }`}>
                        {s.status}
                      </span>
                    </div>
                    <div className="w-full bg-slate-700 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full transition-all duration-500 ${
                          pct > 90 ? 'bg-red-500' :
                          pct > 70 ? 'bg-amber-400' : 'bg-green-500'
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    {s.amenities && s.amenities.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {s.amenities.map((a, j) => (
                          <span key={j} className="px-1.5 py-0.5 bg-slate-700/60 rounded text-xs text-slate-400">
                            {a.replace(/_/g, ' ')}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Displacement flows */}
          {flows.length > 0 && (
            <div>
              <h3 className="text-slate-400 text-xs font-bold uppercase tracking-wider mb-2">
                Displacement Flows
              </h3>
              <div className="space-y-1.5">
                {flows.map((f, i) => (
                  <div key={i} className="flex items-center gap-3 px-3 py-2 bg-slate-800/60 rounded-lg text-sm">
                    <span className="text-slate-200 font-bold flex-1 truncate">{f.origin}</span>
                    <span className="text-indigo-400 font-bold">→</span>
                    <span className="text-slate-400 flex-1 truncate">{f.destination}</span>
                    <span className="text-amber-300 font-bold text-xs">{f.people_count} people</span>
                    <span className={`text-xs font-bold ${
                      f.status === 'SHELTERED' ? 'text-green-400' :
                      f.status === 'IN_TRANSIT' ? 'text-yellow-400' : 'text-slate-500'
                    }`}>
                      {f.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Capacity summary */}
          {summary && (
            <div className="mt-4 text-center">
              <span className="text-slate-600 text-xs">
                Shelter capacity used: {summary.shelter_capacity_used_pct}% |
                Shelters active: {summary.shelters_active} |
                Returned home: {summary.returned_home}
              </span>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default DisplacementMap
