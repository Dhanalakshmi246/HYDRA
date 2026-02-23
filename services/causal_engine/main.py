"""Causal Engine — FastAPI service (port 8007).

Exposes:
  GET  /api/v1/causal/dag           → full DAG graph
  POST /api/v1/causal/predict       → forward inference
  POST /api/v1/causal/intervene     → do(X=x) intervention
  POST /api/v1/causal/sensitivity   → sweep analysis
  GET  /api/v1/causal/adjacency     → adjacency dict
  GET  /health                      → liveness
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, List

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException

from shared.config import get_settings
from shared.models.phase2 import (
    CausalDAG,
    InterventionRequest,
    InterventionResult,
)
from services.causal_engine.dag import load_dag, save_dag
from services.causal_engine.gnn import CausalGNNEngine
from services.causal_engine.interventions import InterventionAPI

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Globals wired at startup ─────────────────────────────────────────────
_engine: CausalGNNEngine | None = None
_api: InterventionAPI | None = None
_dag: CausalDAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _api, _dag
    logger.info("causal_engine_starting", port=settings.CAUSAL_ENGINE_PORT)

    # Load or build the DAG
    _dag = load_dag(settings.CAUSAL_DAG_PATH)
    logger.info("dag_loaded", nodes=len(_dag.nodes), edges=len(_dag.edges))

    # Init GNN engine
    _engine = CausalGNNEngine(
        _dag,
        hidden=settings.CAUSAL_GNN_HIDDEN,
        n_layers=settings.CAUSAL_GNN_LAYERS,
    )
    _api = InterventionAPI(_engine)

    # Persist default DAG if file doesn't exist
    save_dag(_dag, settings.CAUSAL_DAG_PATH)

    logger.info("causal_engine_ready")
    yield
    logger.info("causal_engine_shutdown")


app = FastAPI(
    title="ARGUS Causal Engine",
    version="2.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "service": "causal_engine",
        "version": "2.0.0",
        "status": "healthy",
        "dag_nodes": len(_dag.nodes) if _dag else 0,
        "dag_edges": len(_dag.edges) if _dag else 0,
    }


@app.get("/api/v1/causal/dag")
async def get_dag():
    """Return the full causal DAG."""
    if not _dag:
        raise HTTPException(503, "DAG not loaded")
    return _dag.model_dump()


@app.post("/api/v1/causal/predict")
async def predict(evidence: Dict[str, float]):
    """Forward pass through the causal graph."""
    if not _engine:
        raise HTTPException(503, "Engine not ready")
    result = _engine.predict(evidence)
    return {"evidence": evidence, "predictions": result}


@app.post("/api/v1/causal/intervene")
async def intervene(request: InterventionRequest) -> InterventionResult:
    """Execute a do(X=x) intervention."""
    if not _api:
        raise HTTPException(503, "Engine not ready")
    if request.variable not in _engine.idx:
        raise HTTPException(
            400,
            f"Unknown variable '{request.variable}'. "
            f"Available: {list(_engine.idx.keys())}",
        )
    return _api.run(request)


@app.post("/api/v1/causal/sensitivity")
async def sensitivity(
    variable: str,
    target: str,
    min_val: float = 0.0,
    max_val: float = 1.0,
    steps: int = 10,
    context: Dict[str, float] | None = None,
):
    """Sweep a variable across a range and observe target response."""
    if not _api:
        raise HTTPException(503, "Engine not ready")
    import numpy as np

    values = np.linspace(min_val, max_val, steps).tolist()
    results = _api.sensitivity_analysis(
        variable=variable,
        values=values,
        target=target,
        context=context or {},
    )
    return {"variable": variable, "target": target, "sweep": results}


@app.get("/api/v1/causal/adjacency")
async def adjacency():
    """Return adjacency dictionary."""
    if not _engine:
        raise HTTPException(503, "Engine not ready")
    return _engine.get_adjacency_dict()


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "services.causal_engine.main:app",
        host=settings.SERVICE_HOST,
        port=settings.CAUSAL_ENGINE_PORT,
        reload=False,
    )
