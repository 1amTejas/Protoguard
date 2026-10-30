"""Isolation Forest, implemented in pure Python.

Trained on normal windows only (novelty detection). Anomalies are windows that
random axis-parallel splits isolate quickly. It looks at all features jointly,
so it can catch combinations that no single feature reveals. The trees are
stored as nested lists, so the model serialises to plain JSON.

Caveat: if every training window had identical feature values the trees could
not split and the score would be constant. Real traffic has natural variation,
so this only matters for artificial data with no noise.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Sequence

from ..errors import ModelError
from ..windows import FEATURES, Window, vector
from .base import Detector

EULER_GAMMA = 0.5772156649
MIN_WINDOWS = 20


def _c(n: int) -> float:
    """Average path length of an unsuccessful BST search over n items."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + EULER_GAMMA) - 2.0 * (n - 1) / n


class IsolationForestDetector(Detector):
    name = "iforest"
    min_threshold = 0.5

    def __init__(self, n_trees: int = 100, sample_size: int = 256, seed: int = 0) -> None:
        if n_trees < 1 or sample_size < 2:
            raise ValueError("n_trees must be >= 1 and sample_size >= 2")
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.seed = seed
        self.proto = ""
        self.psi = 0
        self.trees: List[list] = []

    def fit(self, windows: Sequence[Window]) -> None:
        if len(windows) < MIN_WINDOWS:
            raise ModelError(f"iforest: need at least {MIN_WINDOWS} training windows, got {len(windows)}")
        self.proto = windows[0].proto
        rows = [vector(w) for w in windows]
        rng = random.Random(self.seed)
        self.psi = min(self.sample_size, len(rows))
        max_depth = math.ceil(math.log2(max(self.psi, 2)))
        self.trees = [self._build(rng.sample(rows, self.psi), 0, max_depth, rng) for _ in range(self.n_trees)]

    def _build(self, rows: List[List[float]], depth: int, max_depth: int, rng: random.Random) -> list:
        if depth >= max_depth or len(rows) <= 1:
            return [len(rows)]
        dims = len(rows[0])
        spans = [(min(r[j] for r in rows), max(r[j] for r in rows)) for j in range(dims)]
        usable = [j for j in range(dims) if spans[j][0] < spans[j][1]]
        if not usable:
            return [len(rows)]
        j = rng.choice(usable)
        lo, hi = spans[j]
        split = lo + (hi - lo) * rng.random()
        if not lo < split <= hi:
            split = (lo + hi) / 2.0
        left = [r for r in rows if r[j] < split]
        right = [r for r in rows if r[j] >= split]
        if not left or not right:
            return [len(rows)]
        return [j, split, self._build(left, depth + 1, max_depth, rng), self._build(right, depth + 1, max_depth, rng)]

    @staticmethod
    def _path(tree: list, x: Sequence[float]) -> float:
        node, depth = tree, 0
        while len(node) == 4:
            node = node[2] if x[node[0]] < node[1] else node[3]
            depth += 1
        return depth + _c(node[0])

    def score(self, window: Window) -> float:
        if not self.trees:
            raise ModelError("iforest is not fitted")
        x = vector(window)
        mean_path = sum(self._path(t, x) for t in self.trees) / len(self.trees)
        return 2.0 ** (-mean_path / _c(self.psi))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proto": self.proto, "n_trees": self.n_trees, "sample_size": self.sample_size,
            "seed": self.seed, "psi": self.psi, "trees": self.trees,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IsolationForestDetector":
        try:
            det = cls(int(data["n_trees"]), int(data["sample_size"]), int(data["seed"]))
            det.proto = str(data["proto"])
            det.psi = int(data["psi"])
            det.trees = list(data["trees"])
            if det.proto not in FEATURES or det.psi < 2 or not det.trees:
                raise ValueError("bad forest header")
            dims = len(FEATURES[det.proto])
            for tree in det.trees:
                _validate_tree(tree, dims, 0)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelError(f"invalid iforest detector state: {exc}") from exc
        return det


def _validate_tree(node: Any, dims: int, depth: int) -> None:
    if depth > 64 or not isinstance(node, list):
        raise ValueError("malformed tree")
    if len(node) == 1:
        if not isinstance(node[0], int) or node[0] < 0:
            raise ValueError("malformed leaf")
        return
    if len(node) != 4:
        raise ValueError("malformed node")
    feature, split, left, right = node
    if not isinstance(feature, int) or not 0 <= feature < dims:
        raise ValueError("feature index out of range")
    if not isinstance(split, (int, float)) or not math.isfinite(split):
        raise ValueError("bad split value")
    _validate_tree(left, dims, depth + 1)
    _validate_tree(right, dims, depth + 1)
