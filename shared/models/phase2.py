"""Phase 2 shared Pydantic models for ARGUS advanced AI layer.

Covers:
  - Causal Engine (GNN DAG, interventions, counterfactuals)
  - CHORUS community intelligence
  - Federated Learning
  - Evacuation RL
  - MIRROR counterfactual replay
  - FloodLedger blockchain oracle
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════
# Causal Engine
# ══════════════════════════════════════════════════════════════════════════

class CausalNode(BaseModel):
    """A node in the causal DAG."""
    node_id: str
    variable: str            # e.g. "rainfall", "soil_moisture", "water_level"
    station_id: Optional[str] = None
    village_id: Optional[str] = None
    parents: List[str] = Field(default_factory=list)
    children: List[str] = Field(default_factory=list)
    structural_eq: Optional[str] = None   # human-readable structural equation


class CausalEdge(BaseModel):
    """Directed edge in the causal graph with strength."""
    source: str
    target: str
    weight: float = Field(ge=0.0, le=1.0)
    lag_hours: float = 0.0
    mechanism: Optional[str] = None       # "hydrological", "meteorological", "anthropogenic"


class CausalDAG(BaseModel):
    """Complete causal directed acyclic graph."""
    dag_id: str = "beas_brahmaputra_v1"
    nodes: List[CausalNode] = Field(default_factory=list)
    edges: List[CausalEdge] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now())
    version: str = "1.0.0"


class InterventionRequest(BaseModel):
    """do(X=x) intervention query."""
    variable: str                         # variable to intervene on
    value: float                          # fixed value
    target_variables: List[str]           # variables to observe after intervention
    village_id: Optional[str] = None
    context: Dict[str, float] = Field(default_factory=dict)


class InterventionResult(BaseModel):
    """Result of a causal intervention."""
    intervention: InterventionRequest
    original_values: Dict[str, float] = Field(default_factory=dict)
    counterfactual_values: Dict[str, float] = Field(default_factory=dict)
    causal_effects: Dict[str, float] = Field(default_factory=dict)   # ATE per target
    confidence: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class CounterfactualQuery(BaseModel):
    """What-if counterfactual query for MIRROR."""
    query_id: str
    scenario_name: str                    # e.g. "What if rainfall was 50% less?"
    base_event_id: Optional[str] = None   # historical event to replay
    modifications: Dict[str, float] = Field(default_factory=dict)
    village_id: Optional[str] = None


class CounterfactualResult(BaseModel):
    """Result of a counterfactual simulation."""
    query: CounterfactualQuery
    timeline: List[Dict[str, Any]] = Field(default_factory=list)  # time-stepped outcomes
    base_outcome: Dict[str, float] = Field(default_factory=dict)
    modified_outcome: Dict[str, float] = Field(default_factory=dict)
    risk_delta: float = 0.0              # change in peak risk
    lives_impact: Optional[int] = None   # estimated lives affected
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


# ══════════════════════════════════════════════════════════════════════════
# TFT Deep Track prediction
# ══════════════════════════════════════════════════════════════════════════

class TFTPrediction(BaseModel):
    """Temporal Fusion Transformer prediction output."""
    village_id: str
    horizon_hours: int = 24
    timesteps: List[str] = Field(default_factory=list)          # ISO timestamps
    point_forecast: List[float] = Field(default_factory=list)   # risk score per step
    lower_bound: List[float] = Field(default_factory=list)      # 10th percentile
    upper_bound: List[float] = Field(default_factory=list)      # 90th percentile
    attention_weights: Dict[str, float] = Field(default_factory=dict)  # feature importance
    created_at: datetime = Field(default_factory=lambda: datetime.now())


# ══════════════════════════════════════════════════════════════════════════
# CHORUS — Community Intelligence
# ══════════════════════════════════════════════════════════════════════════

class SentimentLevel(str, Enum):
    CALM = "CALM"
    CONCERNED = "CONCERNED"
    ANXIOUS = "ANXIOUS"
    PANIC = "PANIC"


class CommunityReport(BaseModel):
    """A single community intelligence report (WhatsApp / field)."""
    report_id: str
    village_id: str
    source: str = "whatsapp"              # whatsapp, field_worker, voice_call
    message: str = ""
    translated_text: Optional[str] = None # English translation
    language: str = "hi"                  # ISO 639-1
    sentiment: SentimentLevel = SentimentLevel.CALM
    keywords: List[str] = Field(default_factory=list)
    flood_mentioned: bool = False
    water_level_reported: Optional[float] = None  # meters, community-estimated
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class ChorusAggregation(BaseModel):
    """Aggregated community sentiment for a village."""
    village_id: str
    report_count: int = 0
    dominant_sentiment: SentimentLevel = SentimentLevel.CALM
    panic_ratio: float = 0.0             # fraction of PANIC reports
    avg_credibility: float = 0.5
    flood_mention_rate: float = 0.0
    community_risk_boost: float = 0.0    # additional risk signal [0, 0.3]
    top_keywords: List[str] = Field(default_factory=list)
    window_minutes: int = 60
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


# ══════════════════════════════════════════════════════════════════════════
# Federated Learning
# ══════════════════════════════════════════════════════════════════════════

class FederatedRound(BaseModel):
    """A single round of federated learning."""
    round_id: int
    global_model_version: str
    participating_nodes: List[str] = Field(default_factory=list)
    aggregation_method: str = "fedavg"   # fedavg, fedprox, scaffold
    global_loss: Optional[float] = None
    global_accuracy: Optional[float] = None
    privacy_budget_spent: float = 0.0    # ε for differential privacy
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class NodeUpdate(BaseModel):
    """Model update from a federated node."""
    node_id: str
    round_id: int
    num_samples: int = 0
    local_loss: float = 0.0
    local_accuracy: float = 0.0
    gradient_norm: float = 0.0
    clipped: bool = False                # was DP noise clipped?


# ══════════════════════════════════════════════════════════════════════════
# Evacuation RL
# ══════════════════════════════════════════════════════════════════════════

class EvacuationZone(BaseModel):
    """A zone in the evacuation graph."""
    zone_id: str
    village_id: str
    name: str
    population: int = 0
    lat: float = 0.0
    lon: float = 0.0
    elevation_m: float = 0.0
    is_safe_zone: bool = False
    capacity: Optional[int] = None       # safe zone capacity


class EvacuationRoute(BaseModel):
    """An edge in the evacuation graph."""
    route_id: str
    from_zone: str
    to_zone: str
    distance_km: float
    travel_time_min: float
    capacity_persons_hr: int = 500
    road_type: str = "paved"             # paved, unpaved, bridge, boat
    flood_risk: float = 0.0             # probability route is flooded
    is_passable: bool = True


class EvacuationAction(BaseModel):
    """RL agent's recommended evacuation action."""
    action_id: str
    village_id: str
    zone_id: str
    recommended_route: str               # route_id
    priority: int = 1                    # 1=highest
    estimated_travel_min: float = 0.0
    population_to_move: int = 0
    deadline_minutes: float = 60.0       # time window before flooding
    confidence: float = 0.5


class EvacuationPlan(BaseModel):
    """Complete evacuation plan for a village."""
    plan_id: str
    village_id: str
    trigger_level: str = "WARNING"       # alert level that triggered
    risk_score: float = 0.0
    actions: List[EvacuationAction] = Field(default_factory=list)
    total_population: int = 0
    estimated_clear_time_min: float = 0.0
    zones_at_risk: int = 0
    safe_zones_available: int = 0
    rl_reward: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


# ══════════════════════════════════════════════════════════════════════════
# FloodLedger — Blockchain Oracle
# ══════════════════════════════════════════════════════════════════════════

class LedgerEntry(BaseModel):
    """An immutable ledger entry for a flood event."""
    entry_id: str
    block_number: int
    village_id: str
    event_type: str = "prediction"       # prediction, alert, evacuation, verification
    payload: Dict[str, Any] = Field(default_factory=dict)
    data_hash: str = ""                  # SHA-256 of payload
    previous_hash: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    verified: bool = False
    verifier_node: Optional[str] = None


class LedgerChain(BaseModel):
    """Summary of the FloodLedger blockchain."""
    chain_id: str = "argus_flood_ledger_v1"
    length: int = 0
    last_hash: str = ""
    entries_24h: int = 0
    villages_tracked: int = 0
    integrity_verified: bool = True
