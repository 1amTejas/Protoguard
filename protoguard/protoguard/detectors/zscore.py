"""Robust per-feature outlier detector.

Each feature is log-transformed, then compared with the training median using
a robust scale (the larger of the MAD, the standard deviation and a small
floor, so perfectly regular traffic does not produce infinite z-scores). The
window's score is its largest absolute z-score, which also says *which* feature
looked wrong.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from ..errors import ModelError
from ..windows import FEATURES, Window
from .base import Detector
from ..util import mean_std, median

MIN_WINDOWS = 10


class ZScoreDetector(Detector):
    name = "zscore"
    min_threshold = 3.0
    SCALE_FLOOR = 0.05

    def __init__(self) -> None:
        self.proto = ""
        self.stats: Dict[str, Tuple[float, float]] = {}  # feature -> (median, scale)

    def fit(self, windows: Sequence[Window]) -> None:
        if len(windows) < MIN_WINDOWS:
            raise ModelError(f"zscore: need at least {MIN_WINDOWS} training windows, got {len(windows)}")
        self.proto = windows[0].proto
        self.stats = {}
        for name in FEATURES[self.proto]:
            column = [math.log1p(max(0.0, w.features[name])) for w in windows]
            centre = median(column)
            mad = 1.4826 * median([abs(v - centre) for v in column])
            _mean, std = mean_std(column)
            self.stats[name] = (centre, max(mad, std, self.SCALE_FLOOR))

    def _zscores(self, window: Window) -> Dict[str, float]:
        out = {}
        for name, (centre, scale) in self.stats.items():
            out[name] = abs(math.log1p(max(0.0, window.features[name])) - centre) / scale
        return out

    def score(self, window: Window) -> float:
        return max(self._zscores(window).values(), default=0.0)

    def explain(self, window: Window) -> List[str]:
        z = self._zscores(window)
        worst = sorted(z.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return [f"{name}={window.features[name]:.3g} (z={value:.1f})" for name, value in worst if value >= self.min_threshold]

    def to_dict(self) -> Dict[str, Any]:
        return {"proto": self.proto, "stats": {k: [v[0], v[1]] for k, v in self.stats.items()}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZScoreDetector":
        try:
            det = cls()
            det.proto = str(data["proto"])
            det.stats = {str(k): (float(v[0]), float(v[1])) for k, v in data["stats"].items()}
            if det.proto not in FEATURES or set(det.stats) != set(FEATURES[det.proto]):
                raise ValueError("feature names do not match the protocol")
            if any(not math.isfinite(c) or not math.isfinite(s) or s <= 0 for c, s in det.stats.values()):
                raise ValueError("non-finite or non-positive statistics")
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ModelError(f"invalid zscore detector state: {exc}") from exc
        return det
