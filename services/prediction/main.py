"""Prediction Service — FastAPI application.

Consumes feature vectors from ``features.vector.*`` Kafka topics,
runs XGBoost flood-risk classification with SHAP explanations,
and optionally refines with a Physics-Informed Neural Network
(PINN) sensor mesh for spatial interpolation.

Run: ``uvicorn services.prediction.main:app --reload --port 8004``
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, HTTPException, Query

from shared.config import get_settings
from shared.models.prediction import FloodPrediction, PINNSensorReading, AlertPayload

from services.prediction.predictor import FloodPredictor
from services.prediction.explainer import SHAPExplainer
from services.prediction.pinn import PINNMesh
from services.prediction.consumer import start_prediction_consumer
from services.prediction.alert_publisher import AlertPublisher
from services.prediction.prediction_store import PredictionStore

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Singletons ───────────────────────────────────────────────────────────
prediction_store = PredictionStore()
predictor = FloodPredictor()
explainer = SHAPExplainer(predictor)
pinn = PINNMesh()
alert_publisher = AlertPublisher()


# ── Lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start Kafka consumer on startup; cleanup on shutdown."""
    logger.info("prediction_service_starting")
    consumer_task = asyncio.create_task(
        start_prediction_consumer(
            predictor=predictor,
            explainer=explainer,
            pinn=pinn,
            store=prediction_store,
            alert_publisher=alert_publisher,
        )
    )
    yield
    consumer_task.cancel()
    logger.info("prediction_service_stopped")


app = FastAPI(
    title="ARGUS Prediction Service",
    version="1.0.0",
    description="XGBoost flood prediction with SHAP & PINN mesh",
    lifespan=lifespan,
)


# ── Health ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Service health check."""
    return {"status": "ok", "service": "prediction"}


# ── Latest prediction ───────────────────────────────────────────────────
@app.get("/api/v1/prediction/{station_id}/latest", response_model=FloodPrediction)
async def get_latest_prediction(station_id: str):
    """Return the most recent flood prediction for *station_id*."""
    pred = prediction_store.get_latest(station_id)
    if pred is None:
        raise HTTPException(status_code=404, detail=f"No prediction for {station_id}")
    return pred


# ── Prediction history ──────────────────────────────────────────────────
@app.get("/api/v1/prediction/{station_id}/history", response_model=list[FloodPrediction])
async def get_prediction_history(
    station_id: str,
    limit: int = Query(default=50, ge=1, le=500),
):
    """Return recent prediction history for a station."""
    return prediction_store.get_history(station_id, limit)


# ── PINN mesh cell ──────────────────────────────────────────────────────
@app.get("/api/v1/pinn/{grid_cell_id}/latest", response_model=PINNSensorReading)
async def get_pinn_cell(grid_cell_id: str):
    """Return latest PINN interpolated reading for a mesh cell."""
    reading = prediction_store.get_pinn_cell(grid_cell_id)
    if reading is None:
        raise HTTPException(status_code=404, detail=f"No PINN data for {grid_cell_id}")
    return reading


# ── Bulk predictions ────────────────────────────────────────────────────
@app.get("/api/v1/prediction/bulk", response_model=list[FloodPrediction])
async def get_bulk_predictions(
    station_ids: str = Query(..., description="Comma separated station IDs"),
):
    """Bulk fetch latest predictions."""
    ids = [s.strip() for s in station_ids.split(",") if s.strip()]
    return [
        p for sid in ids
        if (p := prediction_store.get_latest(sid)) is not None
    ]


# ── Model metadata ──────────────────────────────────────────────────────
@app.get("/api/v1/model/info")
async def model_info():
    """Return metadata about loaded models."""
    return {
        "xgboost": {
            "version": predictor.model_version,
            "features": predictor.feature_names,
            "loaded": predictor.is_loaded,
        },
        "pinn": {
            "grid_cells": pinn.num_cells,
            "loaded": pinn.is_loaded,
        },
    }
