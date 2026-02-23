"""FloodLedger — FastAPI service (port 8008).

Immutable blockchain for flood event records.

Exposes:
  POST /api/v1/ledger/record           → add an event entry
  GET  /api/v1/ledger/chain            → full chain
  GET  /api/v1/ledger/chain/summary    → chain summary stats
  GET  /api/v1/ledger/block/{number}   → specific block
  GET  /api/v1/ledger/village/{id}     → entries for a village
  GET  /api/v1/ledger/verify           → integrity check
  GET  /health                         → liveness
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import get_settings
from services.flood_ledger.chain import FloodLedger
from services.flood_ledger.store import LedgerStore

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Globals ──────────────────────────────────────────────────────────────
_ledger: FloodLedger | None = None
_store: LedgerStore | None = None


class RecordRequest(BaseModel):
    village_id: str
    event_type: str = "prediction"
    payload: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ledger, _store
    logger.info("flood_ledger_starting", port=settings.LEDGER_PORT)

    _ledger = FloodLedger(difficulty=settings.LEDGER_DIFFICULTY)
    _store = LedgerStore(db_path=settings.LEDGER_DB_PATH)

    # Load persisted chain
    loaded = _store.load_chain(_ledger)
    logger.info("flood_ledger_ready", chain_length=len(_ledger.chain), loaded=loaded)
    yield

    # Persist any remaining blocks
    for block in _ledger.chain:
        _store.save_block(block)
    _store.close()
    logger.info("flood_ledger_shutdown")


app = FastAPI(
    title="ARGUS FloodLedger",
    version="2.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    stats = _store.get_stats() if _store else {}
    return {
        "service": "flood_ledger",
        "version": "2.0.0",
        "status": "healthy",
        "chain_length": len(_ledger.chain) if _ledger else 0,
        **stats,
    }


@app.post("/api/v1/ledger/record")
async def record(req: RecordRequest):
    """Add a new entry to the ledger (auto-mines a block)."""
    if not _ledger or not _store:
        raise HTTPException(503, "Ledger not ready")
    entry = _ledger.add_entry(
        village_id=req.village_id,
        event_type=req.event_type,
        payload=req.payload,
        auto_mine=True,
    )
    # Persist the new block
    _store.save_block(_ledger.chain[-1])
    return {
        "entry_id": entry.entry_id,
        "block_number": _ledger.chain[-1].block_number,
        "hash": _ledger.chain[-1].hash,
        "data_hash": entry.data_hash,
    }


@app.get("/api/v1/ledger/chain")
async def get_chain():
    """Return the full chain."""
    if not _ledger:
        raise HTTPException(503, "Ledger not ready")
    return _ledger.get_full_chain()


@app.get("/api/v1/ledger/chain/summary")
async def chain_summary():
    if not _ledger:
        raise HTTPException(503, "Ledger not ready")
    return _ledger.get_chain_summary().model_dump()


@app.get("/api/v1/ledger/block/{block_number}")
async def get_block(block_number: int):
    if not _ledger:
        raise HTTPException(503, "Ledger not ready")
    block = _ledger.get_block(block_number)
    if block is None:
        raise HTTPException(404, f"Block {block_number} not found")
    return block


@app.get("/api/v1/ledger/village/{village_id}")
async def village_entries(village_id: str):
    if not _ledger:
        raise HTTPException(503, "Ledger not ready")
    entries = _ledger.get_entries_for_village(village_id)
    return [e.model_dump() for e in entries]


@app.get("/api/v1/ledger/verify")
async def verify():
    """Verify chain integrity."""
    if not _ledger:
        raise HTTPException(503, "Ledger not ready")
    valid = _ledger.verify_chain()
    return {
        "integrity": valid,
        "chain_length": len(_ledger.chain),
        "last_hash": _ledger.chain[-1].hash if _ledger.chain else "",
    }


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "services.flood_ledger.main:app",
        host=settings.SERVICE_HOST,
        port=settings.LEDGER_PORT,
        reload=False,
    )
