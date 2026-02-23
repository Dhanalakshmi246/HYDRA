import { useState, useMemo, useCallback } from 'react'
import Map from 'react-map-gl/mapbox'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer, PolygonLayer } from '@deck.gl/layers'
import { HeatmapLayer } from '@deck.gl/aggregation-layers'
import 'mapbox-gl/dist/mapbox-gl.css'

import { riskColorRGBA, riskRadius } from '../utils/colorScale'
import VillagePopup from './VillagePopup'

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || ''

// Initial view — centered between HP and Assam
const INITIAL_VIEW = {
  longitude: 85.5,
  latitude: 29.0,
  zoom: 5.2,
  pitch: 45,
  bearing: -15,
}

// Flood polygon overlay (simulated — Beas river corridor)
const FLOOD_POLYGON = [
  [
    [76.9, 31.65],
    [77.2, 31.65],
    [77.25, 32.25],
    [77.15, 32.3],
    [76.85, 31.75],
    [76.9, 31.65],
  ],
]

export default function ARGUSMap({ predictions }) {
  const [viewState, setViewState] = useState(INITIAL_VIEW)
  const [selectedVillage, setSelectedVillage] = useState(null)

  // ── Deck.gl layers ─────────────────────────────────────
  const layers = useMemo(() => {
    const hasEmergency = predictions.some(
      (p) => p.alert_level === 'WARNING' || p.alert_level === 'EMERGENCY'
    )

    return [
      // Heatmap underlay
      new HeatmapLayer({
        id: 'risk-heatmap',
        data: predictions,
        getPosition: (d) => [d.lon, d.lat],
        getWeight: (d) => d.risk_score,
        radiusPixels: 80,
        intensity: 1.2,
        threshold: 0.1,
        colorRange: [
          [34, 197, 94, 40],
          [234, 179, 8, 80],
          [249, 115, 22, 120],
          [239, 68, 68, 160],
          [220, 38, 38, 200],
        ],
        opacity: 0.4,
      }),

      // Flood polygon (only show during active alerts)
      ...(hasEmergency
        ? [
            new PolygonLayer({
              id: 'flood-polygon',
              data: [{ polygon: FLOOD_POLYGON[0] }],
              getPolygon: (d) => d.polygon,
              getFillColor: [239, 68, 68, 50],
              getLineColor: [239, 68, 68, 150],
              lineWidthMinPixels: 2,
              filled: true,
              stroked: true,
              opacity: 0.6,
            }),
          ]
        : []),

      // Village scatter dots
      new ScatterplotLayer({
        id: 'village-dots',
        data: predictions,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: (d) => riskRadius(d.risk_score),
        getFillColor: (d) => riskColorRGBA(d.risk_score),
        getLineColor: [255, 255, 255, 120],
        lineWidthMinPixels: 1,
        stroked: true,
        filled: true,
        radiusScale: 1,
        radiusUnits: 'pixels',
        pickable: true,
        autoHighlight: true,
        highlightColor: [0, 201, 255, 180],
        onClick: ({ object }) => {
          if (object) setSelectedVillage(object)
        },
        updateTriggers: {
          getRadius: predictions.map((p) => p.risk_score),
          getFillColor: predictions.map((p) => p.risk_score),
        },
      }),
    ]
  }, [predictions])

  const handleClosePopup = useCallback(() => setSelectedVillage(null), [])

  return (
    <div className="relative w-full h-full">
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState: vs }) => setViewState(vs)}
        controller={true}
        layers={layers}
        getCursor={({ isHovering }) => (isHovering ? 'pointer' : 'grab')}
      >
        {MAPBOX_TOKEN && (
          <Map
            mapboxAccessToken={MAPBOX_TOKEN}
            mapStyle="mapbox://styles/mapbox/dark-v11"
            attributionControl={false}
          />
        )}

        {/* Fallback dark background when no Mapbox token */}
        {!MAPBOX_TOKEN && (
          <div className="absolute inset-0 bg-navy" />
        )}
      </DeckGL>

      {/* Village popup */}
      {selectedVillage && (
        <VillagePopup village={selectedVillage} onClose={handleClosePopup} />
      )}

      {/* No-token banner */}
      {!MAPBOX_TOKEN && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-navy-mid/90 backdrop-blur px-4 py-2 rounded-lg border border-accent/30 text-sm text-gray-400 font-body">
          Set <code className="text-accent">VITE_MAPBOX_TOKEN</code> in{' '}
          <code className="text-accent">.env</code> for map tiles
        </div>
      )}
    </div>
  )
}
