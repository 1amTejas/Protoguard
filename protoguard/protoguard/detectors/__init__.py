"""Anomaly detectors and the registry used to (de)serialise models."""
from .base import Detector
from .iforest import IsolationForestDetector
from .markov import MarkovDetector
from .rules import RuleDetector
from .zscore import ZScoreDetector

REGISTRY = {
    "rules": RuleDetector,
    "zscore": ZScoreDetector,
    "markov": MarkovDetector,
    "iforest": IsolationForestDetector,
}

__all__ = [
    "Detector",
    "IsolationForestDetector",
    "MarkovDetector",
    "REGISTRY",
    "RuleDetector",
    "ZScoreDetector",
]
