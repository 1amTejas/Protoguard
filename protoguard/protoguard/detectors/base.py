"""Detector interface.

A detector learns what normal looks like from windows of *normal* traffic and
returns a score per window where higher means more anomalous. Scores are
turned into alerts by thresholds that ``GuardModel`` calibrates on held-out
normal data (see ``model.py``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from ..windows import Window


class Detector(ABC):
    name = "base"
    #: Calibrated thresholds are never allowed below this value.
    min_threshold = 0.0
    #: If set, the threshold is fixed and calibration is skipped (hard rules).
    fixed_threshold: Optional[float] = None

    @abstractmethod
    def fit(self, windows: Sequence[Window]) -> None:
        """Learn from windows of normal traffic (all of one protocol)."""

    @abstractmethod
    def score(self, window: Window) -> float:
        """Anomaly score for one window (higher = more anomalous)."""

    def explain(self, window: Window) -> List[str]:
        """Short human-readable reasons for the window's score."""
        return []

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable state. Models are never pickled."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detector":
        """Rebuild a detector from :meth:`to_dict` output, validating its shape."""
