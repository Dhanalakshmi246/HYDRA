"""CHORUS NLP — Keyword-based sentiment and flood-relevance analysis.

Lightweight classifier that works without heavy NLP models.
Uses keyword dictionaries for Hindi/English flood terminology and
a rule-based sentiment scorer.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import structlog

from shared.models.phase2 import CommunityReport, SentimentLevel

logger = structlog.get_logger(__name__)

# ── Flood keyword dictionaries ──────────────────────────────────────────
FLOOD_KEYWORDS_EN = {
    "flood", "flooding", "water level", "rising water", "overflow",
    "submerged", "inundated", "waterlogged", "embankment", "breach",
    "dam release", "cloudburst", "landslide", "evacuation", "rescue",
    "marooned", "stranded", "washed away", "danger", "emergency",
    "river bank", "overflowing", "heavy rain", "downpour",
}

FLOOD_KEYWORDS_HI = {
    "baadh", "baarish", "paani", "nadi", "ubhar",
    "doob", "behna", "toofan", "barsat", "khatra",
    "bachao", "madad", "tabaahi", "jal", "pralay",
    "seelab", "daldal", "bhaari", "dhara", "kinara",
}

PANIC_KEYWORDS = {
    "help", "emergency", "sos", "bachao", "madad",
    "danger", "dying", "trapped", "rescue", "urgent",
    "khatra", "maut", "fasa", "doob raha",
}

CALM_KEYWORDS = {
    "normal", "safe", "ok", "fine", "sab theek",
    "stable", "receding", "improving", "controlled",
}


def _extract_keywords(text: str) -> List[str]:
    """Extract flood-related keywords from text."""
    text_lower = text.lower()
    found = []
    for kw in FLOOD_KEYWORDS_EN | FLOOD_KEYWORDS_HI:
        if kw in text_lower:
            found.append(kw)
    return found


def _score_sentiment(text: str, keywords: List[str]) -> SentimentLevel:
    """Rule-based sentiment classification."""
    text_lower = text.lower()
    panic_count = sum(1 for kw in PANIC_KEYWORDS if kw in text_lower)
    calm_count = sum(1 for kw in CALM_KEYWORDS if kw in text_lower)
    flood_intensity = len(keywords)

    if panic_count >= 2 or (panic_count >= 1 and flood_intensity >= 3):
        return SentimentLevel.PANIC
    if flood_intensity >= 3 or panic_count >= 1:
        return SentimentLevel.ANXIOUS
    if flood_intensity >= 1:
        return SentimentLevel.CONCERNED
    if calm_count >= 1:
        return SentimentLevel.CALM
    return SentimentLevel.CALM


def _extract_water_level(text: str) -> Optional[float]:
    """Try to extract a numeric water level from text."""
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:meter|metre|m|feet|ft)",
        r"water\s*(?:level|height)?\s*(?:is|at|around|about)?\s*(\d+(?:\.\d+)?)",
        r"paani\s*(\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        match = re.search(pat, text.lower())
        if match:
            val = float(match.group(1))
            # If in feet, convert to meters
            if "feet" in text.lower() or "ft" in text.lower():
                val *= 0.3048
            return round(val, 2)
    return None


def _assess_credibility(
    text: str,
    keywords: List[str],
    source: str,
    has_location: bool,
) -> float:
    """Heuristic credibility score."""
    score = 0.3  # base
    # Longer messages with detail are more credible
    if len(text) > 50:
        score += 0.1
    if len(text) > 150:
        score += 0.1
    # Flood keywords boost credibility
    score += min(0.2, len(keywords) * 0.05)
    # Named source types
    if source in ("field_worker", "government"):
        score += 0.2
    # Location data
    if has_location:
        score += 0.1
    return min(1.0, round(score, 2))


def analyze_message(
    message: str,
    village_id: str,
    source: str = "whatsapp",
    language: str = "hi",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> CommunityReport:
    """Analyze a single community message and return a structured report."""
    keywords = _extract_keywords(message)
    sentiment = _score_sentiment(message, keywords)
    water_level = _extract_water_level(message)
    flood_mentioned = len(keywords) > 0
    credibility = _assess_credibility(
        message, keywords, source, has_location=(lat is not None)
    )

    report = CommunityReport(
        report_id=str(uuid.uuid4()),
        village_id=village_id,
        source=source,
        message=message,
        language=language,
        sentiment=sentiment,
        keywords=keywords,
        flood_mentioned=flood_mentioned,
        water_level_reported=water_level,
        location_lat=lat,
        location_lon=lon,
        credibility_score=credibility,
    )
    logger.info(
        "message_analyzed",
        village=village_id,
        sentiment=sentiment.value,
        keywords=len(keywords),
        flood=flood_mentioned,
        credibility=credibility,
    )
    return report
