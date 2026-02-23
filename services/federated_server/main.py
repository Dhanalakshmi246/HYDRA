"""Federated Learning Server — FastAPI service (port 8010).

Coordinates federated learning across ACN edge nodes with
differential privacy guarantees.

Exposes:
  GET  /api/v1/fl/model               → download global model weights
  POST /api/v1/fl/update               → submit local update
  POST /api/v1/fl/round                → trigger aggregation round
  GET  /api/v1/fl/status               → convergence + round info
  GET  /api/v1/fl/history              → round history
  POST /api/v1/fl/demo/round           → simulate a full round
  GET  /health                         → liveness
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import get_settings
from services.federated_server.aggregator import (
    FederatedAggregator,
    create_synthetic_global_model,
    simulate_client_update,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Globals ──────────────────────────────────────────────────────────────
_aggregator: FederatedAggregator | None = None
_pending_updates: List = []


class ClientUpdate(BaseModel):
    node_id: str
    n_samples: int = 100
    local_loss: float = 0.0
    local_accuracy: float = 0.0
    # Weights encoded as base64 numpy arrays per layer
    weights: Dict[str, str] = {}  # key -> base64-encoded ndarray


def _encode_weights(weights: Dict[str, np.ndarray]) -> Dict[str, str]:
    """Encode numpy arrays to base64 for JSON transport."""
    encoded = {}
    for key, arr in weights.items():
        buf = io.BytesIO()
        np.save(buf, arr)
        encoded[key] = base64.b64encode(buf.getvalue()).decode()
    return encoded


def _decode_weights(encoded: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Decode base64 strings back to numpy arrays."""
    decoded = {}
    for key, b64 in encoded.items():
        buf = io.BytesIO(base64.b64decode(b64))
        decoded[key] = np.load(buf)
    return decoded


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _aggregator
    logger.info("fl_server_starting", port=settings.FL_SERVER_PORT)

    # Init global model
    global_weights = create_synthetic_global_model()
    _aggregator = FederatedAggregator(
        global_weights=global_weights,
        method=settings.FL_AGGREGATION,
        dp_epsilon=settings.FL_DP_EPSILON,
        dp_delta=settings.FL_DP_DELTA,
    )

    logger.info(
        "fl_server_ready",
        method=settings.FL_AGGREGATION,
        dp_epsilon=settings.FL_DP_EPSILON,
    )
    yield
    logger.info("fl_server_shutdown")


app = FastAPI(
    title="ARGUS Federated Learning Server",
    version="2.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "service": "federated_server",
        "version": "2.0.0",
        "status": "healthy",
        "current_round": _aggregator.round_id if _aggregator else 0,
        "pending_updates": len(_pending_updates),
    }


@app.get("/api/v1/fl/model")
async def get_model():
    """Download the current global model weights."""
    if not _aggregator:
        raise HTTPException(503, "Server not ready")
    encoded = _encode_weights(_aggregator.get_global_weights())
    return {
        "round": _aggregator.round_id,
        "version": f"v{_aggregator.round_id}",
        "layers": list(encoded.keys()),
        "weights": encoded,
    }


@app.post("/api/v1/fl/update")
async def submit_update(update: ClientUpdate):
    """Submit a local model update from an edge node."""
    if not _aggregator:
        raise HTTPException(503, "Server not ready")

    if update.weights:
        delta = _decode_weights(update.weights)
    else:
        # If no weights provided, simulate
        delta, _ = simulate_client_update(
            _aggregator.get_global_weights(),
            n_samples=update.n_samples,
        )

    _pending_updates.append((delta, update.n_samples))
    logger.info(
        "client_update_received",
        node=update.node_id,
        samples=update.n_samples,
        pending=len(_pending_updates),
    )
    return {
        "accepted": True,
        "node_id": update.node_id,
        "pending_updates": len(_pending_updates),
        "min_for_round": settings.FL_MIN_CLIENTS,
    }


@app.post("/api/v1/fl/round")
async def trigger_round():
    """Trigger a federated aggregation round."""
    if not _aggregator:
        raise HTTPException(503, "Server not ready")
    if len(_pending_updates) < settings.FL_MIN_CLIENTS:
        return {
            "triggered": False,
            "reason": f"Need at least {settings.FL_MIN_CLIENTS} updates, have {len(_pending_updates)}",
        }
    _aggregator.aggregate(_pending_updates)
    _pending_updates.clear()
    round_info = _aggregator.get_round_info()
    return {
        "triggered": True,
        "round": round_info.model_dump() if round_info else {},
    }


@app.get("/api/v1/fl/status")
async def status():
    if not _aggregator:
        raise HTTPException(503, "Server not ready")
    return _aggregator.get_convergence_metrics()


@app.get("/api/v1/fl/history")
async def history():
    if not _aggregator:
        raise HTTPException(503, "Server not ready")
    return [r.model_dump() for r in _aggregator.history]


@app.post("/api/v1/fl/demo/round")
async def demo_round(n_clients: int = 3, samples_per_client: int = 200):
    """Simulate a complete federated round for demo purposes."""
    if not _aggregator:
        raise HTTPException(503, "Server not ready")

    # Simulate client updates
    updates = []
    for i in range(n_clients):
        delta, n = simulate_client_update(
            _aggregator.get_global_weights(),
            n_samples=samples_per_client,
            noise_scale=0.005,
        )
        updates.append((delta, n))

    _aggregator.aggregate(updates)
    return {
        "round": _aggregator.round_id,
        "clients": n_clients,
        "samples_per_client": samples_per_client,
        "convergence": _aggregator.get_convergence_metrics(),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "services.federated_server.main:app",
        host=settings.SERVICE_HOST,
        port=settings.FL_SERVER_PORT,
        reload=False,
    )
