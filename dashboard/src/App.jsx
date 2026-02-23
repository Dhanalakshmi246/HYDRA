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
 * Root layout — Phase 2 Tab Navigation:
 *
 *  ┌──────────────────── MetricsBar ─────────────────────┐
 *  │ HYDRA ARGUS │ stats...            │ DEMO toggle     │
 *  ├─────────────────────────────────────────────────────┤
 *  │ [🗺 Risk Map] [🚌 Evacuation] [🔮 MIRROR]          │
 *  │ [🔗 FloodLedger] [📢 CHORUS]                       │
 *  ├─────────────────────────────┬───────────────────────┤
 *  │                             │                       │
 *  │     Active Tab Content      │   AlertSidebar        │
 *  │                             │                       │
 *  └─────────────────────────────┴───────────────────────┘
 */

const TABS = [
  { id: 'risk_map', label: 'Risk Map', icon: '🗺️', shortcut: '1' },
  { id: 'evacuation', label: 'Evacuation', icon: '🚌', shortcut: '2' },
  { id: 'mirror', label: 'MIRROR', icon: '🔮', shortcut: '3' },
  { id: 'flood_ledger', label: 'FloodLedger', icon: '🔗', shortcut: '4' },
  { id: 'chorus', label: 'CHORUS', icon: '📢', shortcut: '5' },
]

export default function App() {
  const [demoMode, setDemoMode] = useState(false)
  const [activeTab, setActiveTab] = useState('risk_map')

  const { predictions, loading, stale, lastUpdated, activeAlerts } =
    usePredictions(demoMode)

  const { alerts, loading: alertsLoading } = useAlertLog(demoMode)

  // Compute max risk for Phase 2 panels
  const maxRisk = predictions.length > 0
    ? Math.max(...predictions.map((p) => p.risk_score || 0))
    : 0

  // Keyboard shortcuts for tabs
  if (typeof window !== 'undefined') {
    window.onkeydown = (e) => {
      if (e.altKey && e.key >= '1' && e.key <= '5') {
        e.preventDefault()
        setActiveTab(TABS[parseInt(e.key) - 1].id)
      }
    }
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'risk_map':
        return (
          <div className="flex-1 relative">
            {loading && predictions.length === 0 ? (
              <div className="absolute inset-0 flex items-center justify-center bg-navy">
                <div className="text-center">
                  <div className="w-10 h-10 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                  <p className="text-gray-400 font-body text-sm">Loading predictions...</p>
                </div>
              </div>
            ) : (
              <ARGUSMap predictions={predictions} />
            )}
            <RiskLegend />
            <ACNStatus predictions={predictions} demoMode={demoMode} />
          </div>
        )

      case 'evacuation':
        return (
          <div className="flex-1 relative overflow-auto">
            <EvacuationMap
              selectedVillage="kullu_01"
              riskScore={maxRisk}
              demoMode={demoMode}
              fullScreen={true}
            />
          </div>
        )

      case 'mirror':
        return (
          <div className="flex-1 relative overflow-auto">
            <MirrorPanel demoMode={demoMode} fullScreen={true} />
          </div>
        )

      case 'flood_ledger':
        return (
          <div className="flex-1 relative overflow-auto">
            <FloodLedger demoMode={demoMode} fullScreen={true} />
          </div>
        )

      case 'chorus':
        return (
          <div className="flex-1 relative overflow-auto">
            <ChorusActivity demoMode={demoMode} fullScreen={true} />
          </div>
        )

      default:
        return null
    }
  }

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

      {/* Tab navigation */}
      <div className="flex items-center gap-1 px-4 py-1.5 bg-gray-900/80 border-b border-gray-800">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
              activeTab === tab.id
                ? 'bg-accent/20 text-accent border border-accent/40 shadow-lg shadow-accent/10'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/60 border border-transparent'
            }`}
            title={`Alt+${tab.shortcut}`}
          >
            <span className="text-sm">{tab.icon}</span>
            <span className="hidden sm:inline">{tab.label}</span>
          </button>
        ))}

        {/* Live status indicators on tab bar */}
        <div className="ml-auto flex items-center gap-3 text-[10px] text-gray-500">
          {predictions.length > 0 && (
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              {predictions.length} stations
            </span>
          )}
          {activeAlerts > 0 && (
            <span className="flex items-center gap-1 text-amber-400">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
              {activeAlerts} alerts
            </span>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {renderTabContent()}

        {/* Alert sidebar — always visible */}
        <AlertSidebar alerts={alerts} loading={alertsLoading} />
      </div>
    </div>
  )
}
