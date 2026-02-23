# ARGUS — Shared models package

# ── Phase 1 models ──────────────────────────────────────────────────────
from shared.models.ingestion import GaugeReading, WeatherData, CCTVFrame
from shared.models.cv_gauging import VirtualGaugeReading
from shared.models.feature_engine import FeatureVector, TemporalFeatures, SpatialFeatures
from shared.models.prediction import (
    FloodPrediction,
    SHAPExplanation,
    PINNSensorReading,
    AlertPayload,
    RiskLevel,
)

# ── Phase 2 models ──────────────────────────────────────────────────────
from shared.models.phase2 import (
    CausalNode,
    CausalEdge,
    CausalDAG,
    InterventionRequest,
    InterventionResult,
    CounterfactualQuery,
    CounterfactualResult,
    TFTPrediction,
    CommunityReport,
    ChorusAggregation,
    SentimentLevel,
    FederatedRound,
    NodeUpdate,
    EvacuationZone,
    EvacuationRoute,
    EvacuationAction,
    EvacuationPlan,
    LedgerEntry,
    LedgerChain,
)

__all__ = [
    # Phase 1
    "GaugeReading",
    "WeatherData",
    "CCTVFrame",
    "VirtualGaugeReading",
    "FeatureVector",
    "TemporalFeatures",
    "SpatialFeatures",
    "FloodPrediction",
    "SHAPExplanation",
    "PINNSensorReading",
    "AlertPayload",
    "RiskLevel",
    # Phase 2
    "CausalNode",
    "CausalEdge",
    "CausalDAG",
    "InterventionRequest",
    "InterventionResult",
    "CounterfactualQuery",
    "CounterfactualResult",
    "TFTPrediction",
    "CommunityReport",
    "ChorusAggregation",
    "SentimentLevel",
    "FederatedRound",
    "NodeUpdate",
    "EvacuationZone",
    "EvacuationRoute",
    "EvacuationAction",
    "EvacuationPlan",
    "LedgerEntry",
    "LedgerChain",
]
