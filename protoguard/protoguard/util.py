"""Small numeric helpers (kept dependency-free on purpose)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence, Tuple


def shannon_entropy(data: bytes, limit: int = 256) -> float:
    """Shannon entropy in bits per byte of the first ``limit`` bytes."""
    chunk = data[:limit]
    if not chunk:
        return 0.0
    n = len(chunk)
    return -sum((c / n) * math.log2(c / n) for c in Counter(chunk).values())


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median of an empty sequence")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Population mean and standard deviation; (0, 0) for an empty sequence."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def threshold_for_fpr(scores: Sequence[float], alpha: float) -> float:
    """Pick a threshold so that at most ``alpha`` of ``scores`` lie strictly above it.

    ``scores`` should come from held-out *normal* data, so ``alpha`` is the
    false-positive rate you are willing to accept on similar traffic.
    """
    if not scores:
        raise ValueError("need at least one calibration score")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    ordered = sorted(scores)
    allowed = int(math.floor(alpha * len(ordered)))
    return ordered[len(ordered) - 1 - allowed]
