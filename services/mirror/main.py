"""MIRROR — FastAPI service (port 8012).

Counterfactual replay: "What if?" flood scenario analysis.

Exposes:
  POST /api/v1/mirror/replay           → run a counterfactual scenario
  GET  /api/v1/mirror/presets          → list preset what-if scenarios
  POST /api/v1/mirror/preset/{id}      → run a preset scenario
  GET  /api/v1/mirror/scenario/{id}    → retrieve cached scenario result
  GET  /api/v1/mirror/scenarios        → list all cached scenario IDs
  POST /api/v1/mirror/compare          → compare base vs modified
  GET  /health                         → liveness
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException

from shared.config import get_settings
from shared.models.phase2 import CounterfactualQuery, CounterfactualResult
from services.mirror.simulator import MirrorEngine

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Globals ──────────────────────────────────────────────────────────────
_engine: MirrorEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    logger.info("mirror_starting", port=settings.MIRROR_PORT)
    _engine = MirrorEngine(max_steps=settings.MIRROR_MAX_STEPS)
    # Pre-run preset scenarios for cache warming
    for preset in _engine.get_preset_scenarios():
        _engine.replay(preset)
    logger.info("mirror_ready", presets=len(_engine.get_preset_scenarios()))
    yield
    logger.info("mirror_shutdown")


app = FastAPI(
    title="ARGUS MIRROR — Counterfactual Replay",
    version="2.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "service": "mirror",
        "version": "2.0.0",
        "status": "healthy",
        "cached_scenarios": len(_engine.list_scenarios()) if _engine else 0,
        "max_steps": settings.MIRROR_MAX_STEPS,
    }


@app.post("/api/v1/mirror/replay")
async def replay(query: CounterfactualQuery) -> dict:
    """Run a custom counterfactual scenario."""
    if not _engine:
        raise HTTPException(503, "Engine not ready")
    result = _engine.replay(query)
    return result.model_dump()


@app.get("/api/v1/mirror/presets")
async def list_presets():
    """List available preset what-if scenarios."""
    if not _engine:
        raise HTTPException(503, "Engine not ready")
    presets = _engine.get_preset_scenarios()
    return [
        {
            "query_id": p.query_id,
            "scenario_name": p.scenario_name,
            "modifications": p.modifications,
        }
        for p in presets
    ]


@app.post("/api/v1/mirror/preset/{preset_id}")
async def run_preset(preset_id: str):
    """Run a preset scenario by ID."""
    if not _engine:
        raise HTTPException(503, "Engine not ready")
    # Check if already cached
    cached = _engine.get_cached(preset_id)
    if cached:
        return cached.model_dump()
    # Find and run preset
    for preset in _engine.get_preset_scenarios():
        if preset.query_id == preset_id:
            result = _engine.replay(preset)
            return result.model_dump()
    raise HTTPException(404, f"Preset '{preset_id}' not found")


@app.get("/api/v1/mirror/scenario/{query_id}")
async def get_scenario(query_id: str):
    """Retrieve a cached scenario result."""
    if not _engine:
        raise HTTPException(503, "Engine not ready")
    result = _engine.get_cached(query_id)
    if not result:
        raise HTTPException(404, f"Scenario '{query_id}' not cached")
    return result.model_dump()


@app.get("/api/v1/mirror/scenarios")
async def list_scenarios():
    """List all cached scenario IDs."""
    if not _engine:
        raise HTTPException(503, "Engine not ready")
    return {"scenarios": _engine.list_scenarios()}


@app.post("/api/v1/mirror/compare")
async def compare(
    base_rainfall: float = 20.0,
    modified_rainfall: float = 10.0,
    base_soil: float = 0.6,
    modified_soil: float = 0.6,
    dam_release: float = 50.0,
    steps: int = 24,
):
    """Quick comparison: base vs modified scenario."""
    if not _engine:
        raise HTTPException(503, "Engine not ready")

    base_q = CounterfactualQuery(
        query_id="compare_base",
        scenario_name="Base scenario",
        modifications={
            "base_rainfall_mm_hr": base_rainfall,
            "base_soil_moisture": base_soil,
            "base_dam_release": dam_release,
            "rainfall_factor": 1.0,
        },
    )
    mod_q = CounterfactualQuery(
        query_id="compare_modified",
        scenario_name="Modified scenario",
        modifications={
            "base_rainfall_mm_hr": base_rainfall,
            "rainfall_mm_hr": modified_rainfall,
            "soil_moisture": modified_soil,
            "base_dam_release": dam_release,
        },
    )
    base_r = _engine.replay(base_q)
    mod_r = _engine.replay(mod_q)

    return {
        "base": {
            "peak_risk": base_r.modified_outcome.get("peak_risk"),
            "peak_level_m": base_r.modified_outcome.get("peak_level_m"),
        },
        "modified": {
            "peak_risk": mod_r.modified_outcome.get("peak_risk"),
            "peak_level_m": mod_r.modified_outcome.get("peak_level_m"),
        },
        "risk_delta": round(
            (mod_r.modified_outcome.get("peak_risk", 0) or 0)
            - (base_r.modified_outcome.get("peak_risk", 0) or 0),
            4,
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "services.mirror.main:app",
        host=settings.SERVICE_HOST,
        port=settings.MIRROR_PORT,
        reload=False,
    )
