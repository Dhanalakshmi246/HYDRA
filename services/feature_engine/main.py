"""Feature Engine — FastAPI service.

Consumes raw gauge, weather, and CV readings from Kafka,
computes temporal / spatial feature vectors, and publishes
them to ``features.vector.{station_id}`` for the Prediction service.

Run: ``uvicorn services.feature_engine.main:app --reload --port 8003``
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, HTTPException, Query

from shared.config import get_settings
from shared.models.feature_engine import FeatureVector, SpatialFeatures, TemporalFeatures

from services.feature_engine.store import FeatureStore
from services.feature_engine.temporal import compute_temporal_features
from services.feature_engine.spatial import compute_spatial_features
from services.feature_engine.builder import build_feature_vector
from services.feature_engine.consumer import start_consumers
from services.feature_engine.publisher import FeaturePublisher

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── In-memory stores (replaced by Redis/TimescaleDB in Phase 2) ─────────
feature_store = FeatureStore()
publisher = FeaturePublisher()


# ── lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start Kafka consumers in background on startup; cleanup on shutdown."""
    logger.info("feature_engine_starting")
    consumer_task = asyncio.create_task(start_consumers(feature_store, publisher))
    yield
    consumer_task.cancel()
    logger.info("feature_engine_stopped")


app = FastAPI(
    title="ARGUS Feature Engine",
    version="1.0.0",
    description="Real-time feature engineering for flood prediction",
    lifespan=lifespan,
)


# ── Health ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Service health check."""
    return {"status": "ok", "service": "feature_engine"}


# ── Latest feature vector ───────────────────────────────────────────────
@app.get("/api/v1/features/{station_id}/latest", response_model=FeatureVector)
async def get_latest_features(station_id: str):
    """Return the most-recently computed feature vector for *station_id*."""
    fv = feature_store.get_latest(station_id)
    if fv is None:
        raise HTTPException(status_code=404, detail=f"No features for station {station_id}")
    return fv


# ── Temporal features only ───────────────────────────────────────────────
@app.get("/api/v1/features/{station_id}/temporal", response_model=TemporalFeatures)
async def get_temporal_features(station_id: str):
    """Return latest temporal features for debugging / inspection."""
    fv = feature_store.get_latest(station_id)
    if fv is None:
        raise HTTPException(status_code=404, detail=f"No features for station {station_id}")
    return fv.temporal


# ── Spatial features only ────────────────────────────────────────────────
@app.get("/api/v1/features/{station_id}/spatial", response_model=SpatialFeatures)
async def get_spatial_features(station_id: str):
    """Return latest spatial features for debugging / inspection."""
    fv = feature_store.get_latest(station_id)
    if fv is None:
        raise HTTPException(status_code=404, detail=f"No features for station {station_id}")
    return fv.spatial


# ── Bulk latest features ────────────────────────────────────────────────
@app.get("/api/v1/features/bulk", response_model=list[FeatureVector])
async def get_bulk_features(
    station_ids: str = Query(..., description="Comma-separated station IDs"),
):
    """Return latest features for multiple stations at once."""
    ids = [s.strip() for s in station_ids.split(",") if s.strip()]
    results = []
    for sid in ids:
        fv = feature_store.get_latest(sid)
        if fv is not None:
            results.append(fv)
    return results


# ── Station topology (used by spatial feature builder) ───────────────────
@app.get("/api/v1/topology")
async def get_topology():
    """Return the station topology / adjacency graph."""
    return feature_store.topology
