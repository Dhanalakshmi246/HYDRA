import { useState } from 'react'

// Phase 1 components
import ARGUSMap from './components/ARGUSMap'
import MetricsBar from './components/MetricsBar'
import AlertSidebar from './components/AlertSidebar'
import RiskLegend from './components/RiskLegend'
import ACNStatus from './components/ACNStatus'

// Phase 2 components
import EvacuationMap from './components/EvacuationMap'
import MirrorPanel from './components/MirrorPanel'
import FloodLedger from './components/FloodLedger'
import ChorusActivity from './components/ChorusActivity'

import usePredictions from './hooks/usePredictions'
import useAlertLog from './hooks/useAlertLog'

/**
 * Root layout — Phase 2:
 *
 *  ┌──────────────────── MetricsBar ─────────────────────┐
 *  │ ARGUS │ stats...               │ DEMO toggle        │
 *  ├─────────────────────────────┬───────────────────────┤
 *  │                             │                       │
 *  │       ARGUSMap (3D)         │   AlertSidebar        │
 *  │   ┌─ RiskLegend             │                       │
 *  │   ├─ ACNStatus              │                       │
 *  │   ├─ EvacuationMap          │                       │
 *  │   ├─ ChorusActivity         │                       │
 *  │   ├─ FloodLedger            │                       │
 *  │   └─ MirrorPanel            │                       │
 *  │                             │                       │
 *  └─────────────────────────────┴───────────────────────┘
 */
export default function App() {
  const [demoMode, setDemoMode] = useState(false)

  const { predictions, loading, stale, lastUpdated, activeAlerts } =
    usePredictions(demoMode)

  const { alerts, loading: alertsLoading } = useAlertLog(demoMode)

  // Compute max risk for Phase 2 panels
  const maxRisk = predictions.length > 0
    ? Math.max(...predictions.map((p) => p.risk_score || 0))
    : 0

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

          {/* Phase 1 overlays */}
          <RiskLegend />
          <ACNStatus predictions={predictions} demoMode={demoMode} />

          {/* Phase 2 overlays — left column stack */}
          <div className="absolute top-2 left-2 flex flex-col gap-2 max-h-[calc(100vh-8rem)] overflow-y-auto z-20 pointer-events-auto">
            <EvacuationMap
              selectedVillage="kullu_01"
              riskScore={maxRisk}
              demoMode={demoMode}
            />
            <ChorusActivity demoMode={demoMode} />
          </div>

          {/* Phase 2 overlays — bottom-left stack */}
          <div className="absolute bottom-2 left-2 flex gap-2 z-20 pointer-events-auto">
            <FloodLedger demoMode={demoMode} />
          </div>

          {/* Phase 2 overlays — bottom-right */}
          <div className="absolute bottom-2 right-80 z-20 pointer-events-auto">
            <MirrorPanel demoMode={demoMode} />
          </div>
        </div>

        {/* Alert sidebar */}
        <AlertSidebar alerts={alerts} loading={alertsLoading} />
      </div>
    </div>
  )
}
