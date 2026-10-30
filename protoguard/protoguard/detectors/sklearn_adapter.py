"""Optional scikit-learn baselines on the same features.

These exist so ProtoGuard's own detectors can be compared with standard
novelty-detection models under the *same* protocol: trained on normal windows
only, thresholds calibrated on held-out normal windows. They are not
serialised (the model format is JSON only; pickle is avoided on purpose).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Sequence

from ..errors import ModelError
from ..util import mean_std
from ..windows import Window, vector
from .base import Detector


class SklearnDetector(Detector):
    """Wrap an estimator with ``fit`` and ``score_samples`` (higher = more normal)."""

    # Negated score_samples can be negative (e.g. OneClassSVM), so the default floor of 0 would
    # silently stop the baseline from ever alerting.
    min_threshold = float("-inf")

    def __init__(self, name: str, factory: Callable[[], Any]) -> None:
        self.name = name
        self._factory = factory
        self._model: Any = None
        self._mean: List[float] = []
        self._std: List[float] = []

    def _standardise(self, row: Sequence[float]) -> List[float]:
        return [(v - m) / s for v, m, s in zip(row, self._mean, self._std)]

    def fit(self, windows: Sequence[Window]) -> None:
        if not windows:
            raise ModelError(f"{self.name}: no training windows")
        rows = [vector(w) for w in windows]
        columns = list(zip(*rows))
        stats = [mean_std(list(col)) for col in columns]
        self._mean = [m for m, _s in stats]
        self._std = [s if s > 1e-6 else 1.0 for _m, s in stats]
        self._model = self._factory()
        self._model.fit([self._standardise(r) for r in rows])

    def score(self, window: Window) -> float:
        if self._model is None:
            raise ModelError(f"{self.name} is not fitted")
        value = float(self._model.score_samples([self._standardise(vector(window))])[0])
        return -value if math.isfinite(value) else float("inf")

    def to_dict(self) -> Dict[str, Any]:
        raise ModelError("scikit-learn baselines cannot be saved (evaluation only)")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detector":
        raise ModelError("scikit-learn baselines cannot be loaded")


def baseline_factories(seed: int = 0) -> Dict[str, Callable[[], Detector]]:
    """Factories for three standard baselines. Needs scikit-learn."""
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.svm import OneClassSVM
    except ImportError as exc:
        raise ModelError("baselines need scikit-learn: pip install 'protoguard[ml]'") from exc
    return {
        "sk_iforest": lambda: SklearnDetector("sk_iforest", lambda: IsolationForest(n_estimators=100, random_state=seed)),
        "sk_ocsvm": lambda: SklearnDetector("sk_ocsvm", lambda: OneClassSVM(gamma="scale", nu=0.05)),
        "sk_lof": lambda: SklearnDetector("sk_lof", lambda: LocalOutlierFactor(n_neighbors=20, novelty=True)),
    }
