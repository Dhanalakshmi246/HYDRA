"""Evacuation RL — FastAPI service (port 8011).

Reinforcement learning-based evacuation choreography engine.

Exposes:
  POST /api/v1/evacuation/plan            → generate evacuation plan
  GET  /api/v1/evacuation/graph           → evacuation zone/route graph
  GET  /api/v1/evacuation/graph/{village}  → village-specific graph
  POST /api/v1/evacuation/train           → train RL agent
  GET  /api/v1/evacuation/evaluate        → evaluate current policy
  GET  /api/v1/evacuation/zones/{village} → zones for a village
  GET  /health                            → liveness
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException

from shared.config import get_settings
from services.evacuation_rl.graph import EvacuationGraph
from services.evacuation_rl.agent import EvacuationAgent

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Globals ──────────────────────────────────────────────────────────────
_graph: EvacuationGraph | None = None
_agents: dict[str, EvacuationAgent] = {}


def _get_agent(village_id: str) -> EvacuationAgent:
    """Get or create an agent for a village."""
    if village_id not in _agents:
        _agents[village_id] = EvacuationAgent(
            graph=_graph,
            village_id=village_id,
            model_path=settings.EVAC_MODEL_PATH,
        )
    return _agents[village_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logger.info("evacuation_rl_starting", port=settings.EVAC_RL_PORT)

    # Load or build evacuation graph
    if Path(settings.EVAC_GRAPH_PATH).exists():
        _graph = EvacuationGraph.from_file(settings.EVAC_GRAPH_PATH)
        logger.info("graph_loaded", path=settings.EVAC_GRAPH_PATH)
    else:
        _graph = EvacuationGraph()
        _graph.save(settings.EVAC_GRAPH_PATH)
        logger.info("default_graph_created")

    # Pre-warm agents for known villages
    for vid in ["kullu_01", "majuli_01"]:
        _get_agent(vid)

    logger.info(
        "evacuation_rl_ready",
        zones=len(_graph.zones),
        routes=len(_graph.routes),
    )
    yield
    logger.info("evacuation_rl_shutdown")


app = FastAPI(
    title="ARGUS Evacuation RL Engine",
    version="2.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "service": "evacuation_rl",
        "version": "2.0.0",
        "status": "healthy",
        "zones": len(_graph.zones) if _graph else 0,
        "routes": len(_graph.routes) if _graph else 0,
        "agents": list(_agents.keys()),
    }


@app.post("/api/v1/evacuation/plan")
async def generate_plan(
    village_id: str = "kullu_01",
    risk_score: float = 0.6,
    trigger_level: str = "WARNING",
):
    """Generate an AI-driven evacuation plan."""
    if not _graph:
        raise HTTPException(503, "Service not ready")
    agent = _get_agent(village_id)
    plan = agent.generate_plan(risk_score=risk_score, trigger_level=trigger_level)
    return plan.model_dump()


@app.get("/api/v1/evacuation/graph")
async def get_graph():
    """Return the full evacuation graph."""
    if not _graph:
        raise HTTPException(503, "Service not ready")
    return _graph.to_dict()


@app.get("/api/v1/evacuation/graph/{village_id}")
async def get_village_graph(village_id: str):
    """Return zones and routes for a specific village."""
    if not _graph:
        raise HTTPException(503, "Service not ready")
    zones = [z for z in _graph.zones if z.village_id == village_id]
    routes_from = set()
    for z in zones:
        for r, _ in _graph.get_routes_from(z.zone_id):
            routes_from.add(r.route_id)
    routes = [r for r in _graph.routes if r.route_id in routes_from]
    return {
        "village_id": village_id,
        "zones": [z.model_dump() for z in zones],
        "routes": [r.model_dump() for r in routes],
        "total_population": sum(z.population for z in zones if z.population > 0),
        "safe_capacity": sum(z.capacity or 0 for z in zones if z.is_safe_zone),
    }


@app.get("/api/v1/evacuation/zones/{village_id}")
async def get_zones(village_id: str):
    """Return zones for a village."""
    if not _graph:
        raise HTTPException(503, "Service not ready")
    zones = [z for z in _graph.zones if z.village_id == village_id]
    return [z.model_dump() for z in zones]


@app.post("/api/v1/evacuation/train")
async def train_agent(
    village_id: str = "kullu_01",
    timesteps: int = 10000,
):
    """Train the RL agent."""
    if not _graph:
        raise HTTPException(503, "Service not ready")
    agent = _get_agent(village_id)
    result = agent.train(total_timesteps=timesteps)
    return result


@app.get("/api/v1/evacuation/evaluate")
async def evaluate_agent(
    village_id: str = "kullu_01",
    episodes: int = 10,
):
    """Evaluate current policy."""
    agent = _get_agent(village_id)
    return agent.evaluate(n_episodes=episodes)


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "services.evacuation_rl.main:app",
        host=settings.SERVICE_HOST,
        port=settings.EVAC_RL_PORT,
        reload=False,
    )
