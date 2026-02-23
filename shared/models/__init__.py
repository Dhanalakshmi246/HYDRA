# ARGUS — Shared models package

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

__all__ = [
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
]
