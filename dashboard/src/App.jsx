import { useState } from 'react'

import ARGUSMap from './components/ARGUSMap'
import MetricsBar from './components/MetricsBar'
import AlertSidebar from './components/AlertSidebar'
import RiskLegend from './components/RiskLegend'
import ACNStatus from './components/ACNStatus'

import usePredictions from './hooks/usePredictions'
import useAlertLog from './hooks/useAlertLog'

/**
 * Root layout:
 *
 *  ┌──────────────── MetricsBar ─────────────────┐
 *  │ ARGUS │ stats...        │ DEMO toggle       │
 *  ├────────────────────────────┬────────────────┤
 *  │                            │                │
 *  │       ARGUSMap (3D)        │  AlertSidebar  │
 *  │   + RiskLegend overlay     │                │
 *  │   + ACNStatus overlay      │                │
 *  │                            │                │
 *  └────────────────────────────┴────────────────┘
 */
export default function App() {
  const [demoMode, setDemoMode] = useState(false)

  const { predictions, loading, stale, lastUpdated, activeAlerts } =
    usePredictions(demoMode)

  const { alerts, loading: alertsLoading } = useAlertLog(demoMode)

  return (
    <div className="h-screen w-screen flex flex-col bg-navy overflow-hidden">
      {/* Top bar */}
      <MetricsBar
        predictions={predictions}
        activeAlerts={activeAlerts}
        lastUpdated={lastUpdated}
        stale={stale}
        demoMode={demoMode}
        onToggleDemo={() => setDemoMode((prev) => !prev)}
      />

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Map area */}
        <div className="flex-1 relative">
          {loading && predictions.length === 0 ? (
            <div className="absolute inset-0 flex items-center justify-center bg-navy">
              <div className="text-center">
                <div className="w-10 h-10 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                <p className="text-gray-400 font-body text-sm">
                  Loading predictions...
                </p>
              </div>
            </div>
          ) : (
            <ARGUSMap predictions={predictions} />
          )}

          {/* Overlays */}
          <RiskLegend />
          <ACNStatus predictions={predictions} demoMode={demoMode} />
        </div>

        {/* Alert sidebar */}
        <AlertSidebar alerts={alerts} loading={alertsLoading} />
      </div>
    </div>
  )
}
