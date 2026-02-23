"""Prediction Service — FastAPI application.

Dual-track prediction engine:
  1. Fast track: XGBoost flood-probability predictor + SHAP explainability
  2. Adaptive alert thresholds (NDMA-based, condition-adjusted)

Consumes enriched features from TimescaleDB (polled every 60s),
runs the prediction pipeline, publishes results to Kafka + Redis,
and exposes REST API endpoints for the dashboard and alert dispatcher.

Also retains the original Kafka ``features.vector.*`` consumer for
backward compatibility with the feature engine.

Run: ``uvicorn services.prediction.main:app --reload --port 8004``
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import structlog
from fastapi import FastAPI, HTTPException, Query

from shared.config import get_settings
from shared.models.prediction import FloodPrediction, PINNSensorReading, AlertPayload

# Original modules (kept for backward compat)
from services.prediction.predictor import FloodPredictor
from services.prediction.explainer import SHAPExplainer
from services.prediction.pinn import PINNMesh
from services.prediction.consumer import start_prediction_consumer
from services.prediction.alert_publisher import AlertPublisher
from services.prediction.prediction_store import PredictionStore

# New fast-track modules
from services.prediction.fast_track.xgboost_predictor import XGBoostPredictor
from services.prediction.fast_track.shap_explainer import SHAPExplainerV2
from services.prediction.fast_track.threshold_engine import ThresholdEngine
from services.prediction.fast_track.alert_classifier import AlertClassifier, AlertLevel, ConfidenceBand
from services.prediction.consumers.feature_consumer import FeatureConsumer
from services.prediction.publishers.prediction_publisher import PredictionPublisher

logger = structlog.get_logger(__name__)
settings = get_settings()

# ══════════════════════════════════════════════════════════════════════════
# Singletons
# ══════════════════════════════════════════════════════════════════════════

# Original prediction pipeline (backward compat)
prediction_store = PredictionStore()
predictor = FloodPredictor()
explainer = SHAPExplainer(predictor)
pinn = PINNMesh()
alert_publisher = AlertPublisher()

# New fast-track prediction pipeline
xgb_predictor = XGBoostPredictor(
    model_path=os.getenv("XGBOOST_MODEL_PATH", "./models/xgboost_flood.joblib"),
    training_data_path=os.getenv("TRAINING_DATA_PATH", "./data/cwc_historical_2019_2023.csv"),
    train_on_startup=os.getenv("TRAIN_ON_STARTUP", "true").lower() in ("true", "1", "yes"),
)
shap_explainer = SHAPExplainerV2(model=xgb_predictor.model)
threshold_engine = ThresholdEngine()
alert_classifier = AlertClassifier()
prediction_publisher = PredictionPublisher()

# In-memory prediction cache (village_id → latest prediction dict)
_predictions_cache: Dict[str, Dict[str, Any]] = {}
_predictions_history: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
    lambda: deque(maxlen=200)
)

# TimescaleDB DSN
_TIMESCALE_DSN = os.getenv(
    "TIMESCALE_URL",
    os.getenv("TIMESCALEDB_DSN", "postgresql://argus:argus@localhost:5432/argus"),
)


# ══════════════════════════════════════════════════════════════════════════
# Core prediction pipeline
# ══════════════════════════════════════════════════════════════════════════

def run_prediction_pipeline(
    village_id: str,
    features: Dict[str, float],
    quality: str = "GOOD",
) -> Dict[str, Any]:
    """Execute the full fast-track prediction pipeline for one village.

    Steps:
      1. XGBoost → flood probability
      2. SHAP → top-3 explanation factors
      3. Adaptive thresholds (soil moisture, monsoon, AMI)
      4. Alert classification
      5. Cache + publish

    Returns:
        Prediction result dict matching the API contract.
    """
    now = datetime.now(timezone.utc)

    # 1. XGBoost prediction
    risk_score = xgb_predictor.predict(features)

    # 2. SHAP explanation
    shap_factors = shap_explainer.explain(features)
    explanation = [f.to_dict() for f in shap_factors]

    # 3. Adaptive thresholds
    soil_moisture = features.get("soil_moisture_index", 0.0)
    ami = features.get("antecedent_moisture_index", 0.0)
    is_monsoon = features.get("is_monsoon_season", 0.0) >= 0.5

    thresholds = threshold_engine.compute(
        soil_moisture_index=soil_moisture,
        is_monsoon_season=is_monsoon,
        antecedent_moisture_index=ami,
    )

    # 4. Alert classification
    alert_level, confidence = alert_classifier.classify(risk_score, thresholds)

    # 5. Build result
    result: Dict[str, Any] = {
        "village_id": village_id,
        "risk_score": round(risk_score, 4),
        "alert_level": alert_level.value,
        "explanation": explanation,
        "adaptive_threshold": {
            "advisory": thresholds.advisory,
            "watch": thresholds.watch,
            "warning": thresholds.warning,
            "emergency": thresholds.emergency,
            "adjustment_reason": thresholds.adjustment_reason,
        },
        "timestamp": now.isoformat(),
        "confidence": confidence.value,
        "quality": quality,
    }

    # Cache
    _predictions_cache[village_id] = result
    _predictions_history[village_id].append(result)

    # Publish to Kafka + Redis
    prediction_publisher.publish(village_id, result)

    logger.info(
        "prediction_complete",
        village=village_id,
        risk_score=risk_score,
        alert_level=alert_level.value,
        confidence=confidence.value,
    )

    return result


# ── Feature consumer callback ────────────────────────────────────────────

def _on_features_received(village_id: str, features: Dict[str, float], quality: str) -> None:
    """Callback invoked by FeatureConsumer for each village."""
    try:
        run_prediction_pipeline(village_id, features, quality)
    except Exception as exc:
        logger.exception("prediction_pipeline_error", village=village_id, error=str(exc))


# ══════════════════════════════════════════════════════════════════════════
# Lifespan
# ══════════════════════════════════════════════════════════════════════════

# Feature consumer instance
feature_consumer = FeatureConsumer(dsn=_TIMESCALE_DSN, on_features=_on_features_received)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start all consumers and background tasks on startup."""
    logger.info("prediction_service_starting", version="2.0.0")

    # Start original Kafka consumer (backward compat)
    kafka_task = asyncio.create_task(
        start_prediction_consumer(
            predictor=predictor,
            explainer=explainer,
            pinn=pinn,
            store=prediction_store,
            alert_publisher=alert_publisher,
        )
    )

    # Start TimescaleDB feature consumer (new fast-track path)
    feature_task = asyncio.create_task(feature_consumer.start())

    yield

    # Shutdown
    kafka_task.cancel()
    await feature_consumer.stop()
    feature_task.cancel()
    prediction_publisher.flush()
    logger.info("prediction_service_stopped")


app = FastAPI(
    title="ARGUS Prediction Service",
    version="2.0.0",
    description=(
        "Dual-track flood prediction: XGBoost fast-path with SHAP explainability "
        "+ adaptive alert thresholds. Consumed by dashboard and alert dispatcher."
    ),
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════════════════
# API Endpoints — primary (used by Dhana's dashboard + alert dispatcher)
# ══════════════════════════════════════════════════════════════════════════


@app.get("/api/v1/health")
async def health_v1():
    """Detailed health check."""
    return {
        "status": "ok",
        "service": "prediction",
        "version": "2.0.0",
        "components": {
            "xgboost_loaded": xgb_predictor.is_loaded,
            "xgboost_model_version": xgb_predictor.model_version,
            "shap_ready": shap_explainer._explainer is not None,
            "feature_consumer_connected": feature_consumer.is_connected,
            "villages_tracked": len(_predictions_cache),
            "original_predictor_loaded": predictor.is_loaded,
            "pinn_loaded": pinn.is_loaded,
        },
    }


@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "ok", "service": "prediction"}


# ── Per-village prediction (Dhana's primary endpoint) ────────────────────
@app.get("/api/v1/prediction/{village_id}")
async def get_prediction(village_id: str):
    """Return the latest prediction for a village.

    Response::

        {
            "village_id": str,
            "risk_score": float,          # 0.0–1.0
            "alert_level": str,           # NORMAL/ADVISORY/WATCH/WARNING/EMERGENCY
            "explanation": [...],         # SHAP top 3
            "adaptive_threshold": {
                "watch": float,
                "warning": float,
                "adjustment_reason": str
            },
            "timestamp": datetime,
            "confidence": str             # LOW/MEDIUM/HIGH
        }
    """
    cached = _predictions_cache.get(village_id)
    if cached is not None:
        return cached

    # Fall back to original station-based prediction store
    pred = prediction_store.get_latest(village_id)
    if pred is not None:
        return pred

    raise HTTPException(
        status_code=404,
        detail=f"No prediction available for village {village_id}",
    )


# ── All predictions (for dashboard map) ──────────────────────────────────
@app.get("/api/v1/predictions/all")
async def get_all_predictions():
    """Return latest predictions for ALL villages (dashboard map rendering)."""
    results = list(_predictions_cache.values())

    # Also include original station-based predictions not yet in cache
    for sid in prediction_store._latest:
        if sid not in _predictions_cache:
            pred = prediction_store.get_latest(sid)
            if pred:
                results.append(pred.model_dump(mode="json"))

    return results


# ── Prediction history ──────────────────────────────────────────────────
@app.get("/api/v1/prediction/{village_id}/history")
async def get_prediction_history_v2(
    village_id: str,
    limit: int = Query(default=50, ge=1, le=500),
):
    """Return recent prediction history for a village."""
    hist = _predictions_history.get(village_id)
    if hist:
        return list(hist)[-limit:]

    # Fall back to original store
    return prediction_store.get_history(village_id, limit)


# ══════════════════════════════════════════════════════════════════════════
# API Endpoints — backward compatible (original pipeline)
# ══════════════════════════════════════════════════════════════════════════


@app.get("/api/v1/prediction/{station_id}/latest", response_model=FloodPrediction)
async def get_latest_prediction_legacy(station_id: str):
    """Return the most recent flood prediction for *station_id* (legacy)."""
    pred = prediction_store.get_latest(station_id)
    if pred is None:
        raise HTTPException(status_code=404, detail=f"No prediction for {station_id}")
    return pred


@app.get("/api/v1/pinn/{grid_cell_id}/latest", response_model=PINNSensorReading)
async def get_pinn_cell(grid_cell_id: str):
    """Return latest PINN interpolated reading for a mesh cell."""
    reading = prediction_store.get_pinn_cell(grid_cell_id)
    if reading is None:
        raise HTTPException(status_code=404, detail=f"No PINN data for {grid_cell_id}")
    return reading


@app.get("/api/v1/prediction/bulk", response_model=list[FloodPrediction])
async def get_bulk_predictions(
    station_ids: str = Query(..., description="Comma separated station IDs"),
):
    """Bulk fetch latest predictions (legacy)."""
    ids = [s.strip() for s in station_ids.split(",") if s.strip()]
    return [
        p for sid in ids
        if (p := prediction_store.get_latest(sid)) is not None
    ]


# ── Model metadata ──────────────────────────────────────────────────────
@app.get("/api/v1/model/info")
async def model_info():
    """Return metadata about all loaded models."""
    return {
        "xgboost_fast_track": {
            "version": xgb_predictor.model_version,
            "features": xgb_predictor.feature_names,
            "loaded": xgb_predictor.is_loaded,
            "train_metrics": xgb_predictor.train_metrics,
        },
        "xgboost_legacy": {
            "version": predictor.model_version,
            "features": predictor.feature_names,
            "loaded": predictor.is_loaded,
        },
        "pinn": {
            "grid_cells": pinn.num_cells,
            "loaded": pinn.is_loaded,
        },
        "threshold_engine": {
            "base_thresholds": threshold_engine._base,
        },
    }


# ── Manual prediction trigger (for testing) ─────────────────────────────
@app.post("/api/v1/prediction/{village_id}/run")
async def trigger_prediction(village_id: str, features: Dict[str, float]):
    """Manually trigger a prediction with provided features (for testing)."""
    result = run_prediction_pipeline(village_id, features)
    return result
