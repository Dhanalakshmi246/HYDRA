"""Centralised configuration for all ARGUS services.

Loads values from environment variables (via ``python-dotenv``)
so that every micro-service shares the same config surface area.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)


class Settings:
    """Simple settings object — reads from env vars with sensible defaults."""

    # ── Kafka ────────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_GROUP_PREFIX: str = os.getenv("KAFKA_GROUP_PREFIX", "argus")

    # ── External APIs ────────────────────────────────────────
    CWC_API_KEY: str = os.getenv("CWC_API_KEY", "")
    IMD_API_KEY: str = os.getenv("IMD_API_KEY", "")

    # ── File paths ───────────────────────────────────────────
    CCTV_REGISTRY_PATH: str = os.getenv("CCTV_REGISTRY_PATH", "./data/cctv_registry.json")
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "./models/yolo11n.pt")
    SAM_MODEL_PATH: str = os.getenv("SAM_MODEL_PATH", "./models/sam2_tiny.pt")
    DEMO_VIDEO_PATH: str = os.getenv("DEMO_VIDEO_PATH", "./data/himachal_flood_demo.mp4")

    # ── CV Gauging ───────────────────────────────────────────
    CV_CONFIDENCE_THRESHOLD: float = float(os.getenv("CV_CONFIDENCE_THRESHOLD", "0.4"))
    PIXEL_TO_METER_DEFAULT: float = float(os.getenv("PIXEL_TO_METER_DEFAULT", "0.05"))

    # ── Prediction ───────────────────────────────────────────
    XGBOOST_MODEL_PATH: str = os.getenv("XGBOOST_MODEL_PATH", "./models/xgboost_flood.joblib")
    PINN_MODEL_PATH: str = os.getenv("PINN_MODEL_PATH", "./models/pinn_mesh_v1.pt")
    PREDICTION_INTERVAL_S: int = int(os.getenv("PREDICTION_INTERVAL_S", "300"))
    SHAP_TOP_K: int = int(os.getenv("SHAP_TOP_K", "5"))
    SHAP_TOP_N: int = int(os.getenv("SHAP_TOP_N", "3"))
    TRAIN_ON_STARTUP: bool = os.getenv("TRAIN_ON_STARTUP", "true").lower() in ("true", "1", "yes")
    FEATURE_POLL_INTERVAL_SEC: int = int(os.getenv("FEATURE_POLL_INTERVAL_SEC", "60"))

    # ── Redis ────────────────────────────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    PREDICTION_REDIS_TTL: int = int(os.getenv("PREDICTION_REDIS_TTL", "300"))

    # ── Feature Engine ───────────────────────────────────────
    FEATURE_WINDOW_1H: int = int(os.getenv("FEATURE_WINDOW_1H", "12"))   # 12 × 5 min = 1 h
    FEATURE_WINDOW_3H: int = int(os.getenv("FEATURE_WINDOW_3H", "36"))
    FEATURE_WINDOW_6H: int = int(os.getenv("FEATURE_WINDOW_6H", "72"))
    FEATURE_WINDOW_24H: int = int(os.getenv("FEATURE_WINDOW_24H", "288"))
    # ── TimescaleDB ──────────────────────────────────────
    TIMESCALEDB_DSN: str = os.getenv(
        "TIMESCALEDB_DSN", "postgresql://argus:argus@localhost:5432/argus"
    )

    # ── PINN ─────────────────────────────────────────────
    PINN_CHECKPOINT_PATH: str = os.getenv("PINN_CHECKPOINT_PATH", "./models/pinn_beas_river.pt")
    # ── Alert Dispatcher ─────────────────────────────────────
    ALERT_KAFKA_TOPIC: str = os.getenv("ALERT_KAFKA_TOPIC", "alerts.dispatch")
    ALERT_COOLDOWN_S: int = int(os.getenv("ALERT_COOLDOWN_S", "900"))  # 15 min

    # ═══════════════════════  PHASE 2  ════════════════════════

    # ── Causal Engine ────────────────────────────────────────
    CAUSAL_DAG_PATH: str = os.getenv("CAUSAL_DAG_PATH", "./shared/causal_dag/beas_brahmaputra_v1.json")
    CAUSAL_GNN_HIDDEN: int = int(os.getenv("CAUSAL_GNN_HIDDEN", "64"))
    CAUSAL_GNN_LAYERS: int = int(os.getenv("CAUSAL_GNN_LAYERS", "3"))
    CAUSAL_ENGINE_PORT: int = int(os.getenv("CAUSAL_ENGINE_PORT", "8007"))

    # ── FloodLedger ──────────────────────────────────────────
    LEDGER_DB_PATH: str = os.getenv("LEDGER_DB_PATH", "./data/flood_ledger.db")
    LEDGER_PORT: int = int(os.getenv("LEDGER_PORT", "8008"))
    LEDGER_DIFFICULTY: int = int(os.getenv("LEDGER_DIFFICULTY", "2"))  # PoW leading zeros

    # ── CHORUS ───────────────────────────────────────────────
    CHORUS_PORT: int = int(os.getenv("CHORUS_PORT", "8009"))
    CHORUS_WINDOW_MIN: int = int(os.getenv("CHORUS_WINDOW_MIN", "60"))
    CHORUS_CREDIBILITY_THRESHOLD: float = float(os.getenv("CHORUS_CREDIBILITY_THRESHOLD", "0.3"))
    WHATSAPP_WEBHOOK_SECRET: str = os.getenv("WHATSAPP_WEBHOOK_SECRET", "")

    # ── Federated Learning ───────────────────────────────────
    FL_SERVER_PORT: int = int(os.getenv("FL_SERVER_PORT", "8010"))
    FL_ROUNDS: int = int(os.getenv("FL_ROUNDS", "10"))
    FL_MIN_CLIENTS: int = int(os.getenv("FL_MIN_CLIENTS", "2"))
    FL_AGGREGATION: str = os.getenv("FL_AGGREGATION", "fedavg")
    FL_DP_EPSILON: float = float(os.getenv("FL_DP_EPSILON", "1.0"))
    FL_DP_DELTA: float = float(os.getenv("FL_DP_DELTA", "1e-5"))

    # ── Evacuation RL ────────────────────────────────────────
    EVAC_RL_PORT: int = int(os.getenv("EVAC_RL_PORT", "8011"))
    EVAC_MODEL_PATH: str = os.getenv("EVAC_MODEL_PATH", "./models/evac_ppo.zip")
    EVAC_GRAPH_PATH: str = os.getenv("EVAC_GRAPH_PATH", "./data/evacuation_graph.json")
    EVAC_RETRAIN_INTERVAL_S: int = int(os.getenv("EVAC_RETRAIN_INTERVAL_S", "3600"))

    # ── MIRROR ───────────────────────────────────────────────
    MIRROR_PORT: int = int(os.getenv("MIRROR_PORT", "8012"))
    MIRROR_MAX_STEPS: int = int(os.getenv("MIRROR_MAX_STEPS", "48"))  # max replay steps

    # ── General ──────────────────────────────────────────────
    SERVICE_HOST: str = os.getenv("SERVICE_HOST", "0.0.0.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a singleton Settings instance."""
    return Settings()
