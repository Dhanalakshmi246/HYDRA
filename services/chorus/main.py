"""CHORUS — FastAPI service (port 8009).

Community intelligence via WhatsApp/field-worker message analysis.

Exposes:
  POST /api/v1/chorus/report            → submit a new community report
  POST /api/v1/chorus/webhook           → WhatsApp webhook endpoint
  GET  /api/v1/chorus/village/{id}      → aggregated sentiment for a village
  GET  /api/v1/chorus/villages          → all village aggregations
  GET  /api/v1/chorus/stats             → service-level stats
  POST /api/v1/chorus/demo/generate     → generate demo reports
  GET  /health                          → liveness
"""

from __future__ import annotations

import random
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from shared.config import get_settings
from services.chorus.nlp import analyze_message
from services.chorus.aggregator import ChorusAggregator

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Globals ──────────────────────────────────────────────────────────────
_aggregator: ChorusAggregator | None = None


class ReportRequest(BaseModel):
    message: str
    village_id: str
    source: str = "whatsapp"
    language: str = "hi"
    lat: Optional[float] = None
    lon: Optional[float] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _aggregator
    logger.info("chorus_starting", port=settings.CHORUS_PORT)
    _aggregator = ChorusAggregator(window_minutes=settings.CHORUS_WINDOW_MIN)
    logger.info("chorus_ready")
    yield
    logger.info("chorus_shutdown")


app = FastAPI(
    title="ARGUS CHORUS — Community Intelligence",
    version="2.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "service": "chorus",
        "version": "2.0.0",
        "status": "healthy",
        "total_reports": _aggregator.get_report_count() if _aggregator else 0,
    }


@app.post("/api/v1/chorus/report")
async def submit_report(req: ReportRequest):
    """Analyze and store a community report."""
    if not _aggregator:
        raise HTTPException(503, "Service not ready")
    report = analyze_message(
        message=req.message,
        village_id=req.village_id,
        source=req.source,
        language=req.language,
        lat=req.lat,
        lon=req.lon,
    )
    if report.credibility_score < settings.CHORUS_CREDIBILITY_THRESHOLD:
        return {
            "accepted": False,
            "reason": "Below credibility threshold",
            "score": report.credibility_score,
        }
    _aggregator.add_report(report)
    return {
        "accepted": True,
        "report_id": report.report_id,
        "sentiment": report.sentiment.value,
        "flood_mentioned": report.flood_mentioned,
        "credibility": report.credibility_score,
        "keywords": report.keywords,
    }


@app.post("/api/v1/chorus/webhook")
async def whatsapp_webhook(request: Request):
    """WhatsApp Business API webhook receiver."""
    body = await request.json()
    # Extract message from WhatsApp webhook format
    try:
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])

        results = []
        for msg in messages:
            text = msg.get("text", {}).get("body", "")
            from_number = msg.get("from", "unknown")
            if text:
                report = analyze_message(
                    message=text,
                    village_id="auto_detect",  # would be resolved via location
                    source="whatsapp",
                    language="hi",
                )
                if report.credibility_score >= settings.CHORUS_CREDIBILITY_THRESHOLD:
                    _aggregator.add_report(report)
                    results.append(report.report_id)

        return {"processed": len(results), "report_ids": results}
    except Exception as e:
        logger.error("webhook_error", error=str(e))
        return {"processed": 0, "error": str(e)}


@app.get("/api/v1/chorus/village/{village_id}")
async def village_aggregation(village_id: str):
    if not _aggregator:
        raise HTTPException(503, "Service not ready")
    agg = _aggregator.aggregate(village_id)
    return agg.model_dump()


@app.get("/api/v1/chorus/villages")
async def all_villages():
    if not _aggregator:
        raise HTTPException(503, "Service not ready")
    return {k: v.model_dump() for k, v in _aggregator.aggregate_all().items()}


@app.get("/api/v1/chorus/stats")
async def stats():
    if not _aggregator:
        raise HTTPException(503, "Service not ready")
    all_agg = _aggregator.aggregate_all()
    return {
        "total_reports": _aggregator.get_report_count(),
        "villages_reporting": len(all_agg),
        "aggregations": {k: v.model_dump() for k, v in all_agg.items()},
    }


@app.post("/api/v1/chorus/demo/generate")
async def demo_generate(village_id: str = "kullu_01", count: int = 10):
    """Generate synthetic demo reports."""
    if not _aggregator:
        raise HTTPException(503, "Service not ready")

    demo_messages = [
        "River water level rising fast near the bridge. Very dangerous!",
        "Heavy rain for 3 hours non-stop. Paani bahut badh raha hai.",
        "Baarish ruk gayi hai. Situation seems stable now.",
        "Bachao! Water entering houses in lower colony. Need rescue boats!",
        "Dam release se paani 2 meter upar aa gaya. SOS!",
        "Fields submerged, road to hospital washed away. Emergency!",
        "Sab theek hai, water level normal. No flooding in our area.",
        "Cloudburst reported upstream. Heavy flooding expected in 3 hours.",
        "Landslide near bridge blocked evacuation route. Khatra!",
        "Nadi ka paani 4 feet upar. Madad chahiye urgently!",
        "Water receding slowly. Situation improving compared to yesterday.",
        "Embankment breach near village school. Children stranded!",
    ]

    generated = []
    for i in range(count):
        msg = random.choice(demo_messages)
        report = analyze_message(
            message=msg,
            village_id=village_id,
            source=random.choice(["whatsapp", "field_worker", "voice_call"]),
            language=random.choice(["hi", "en"]),
        )
        _aggregator.add_report(report)
        generated.append(report.report_id)

    return {
        "generated": len(generated),
        "village_id": village_id,
        "aggregation": _aggregator.aggregate(village_id).model_dump(),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "services.chorus.main:app",
        host=settings.SERVICE_HOST,
        port=settings.CHORUS_PORT,
        reload=False,
    )
