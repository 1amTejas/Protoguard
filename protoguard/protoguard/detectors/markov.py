"""Message-order detector: a first-order Markov chain over message types.

Control traffic is repetitive and ordered (request, response, request, ...).
The chain learns which message type normally follows which. A window is
scored by the average surprise (negative log-likelihood) of its transitions,
so unanswered requests, odd function codes and out-of-order commands stand out
even when the overall rate looks normal. It cannot see *how many* messages
there are, which is what the statistical detectors are for.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Set

from ..errors import ModelError
from ..messages import REQUEST, Message
from ..windows import Window
from .base import Detector

START = "<s>"
UNK = "<unk>"


def message_symbol(m: Message) -> str:
    """Compact symbol for a message: protocol-specific type plus direction."""
    if not m.valid:
        return "invalid"
    if m.proto == "modbus":
        if m.exception is not None:
            return f"e{m.function}"
        return f"{'q' if m.direction == REQUEST else 's'}{m.function}"
    prefix = "c:" if m.direction == REQUEST else "s:"
    if m.kind == "PUBLISH":
        return f"{prefix}PUBLISH{m.qos}"
    return prefix + m.kind


class MarkovDetector(Detector):
    name = "markov"
    min_threshold = 0.0

    def __init__(self, smoothing: float = 0.1) -> None:
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.smoothing = smoothing
        self.proto = ""
        self.counts: Dict[str, Dict[str, int]] = {}
        self._totals: Dict[str, int] = {}
        self._targets: Set[str] = set()

    def fit(self, windows: Sequence[Window]) -> None:
        if not windows:
            raise ModelError("markov: no training windows")
        self.proto = windows[0].proto
        counts: Dict[str, Counter] = defaultdict(Counter)
        for window in windows:
            previous = START
            for m in window.messages:
                symbol = message_symbol(m)
                counts[previous][symbol] += 1
                previous = symbol
        self._load({a: dict(c) for a, c in counts.items()})

    def _load(self, counts: Dict[str, Dict[str, int]]) -> None:
        self.counts = counts
        self._totals = {a: sum(row.values()) for a, row in counts.items()}
        self._targets = {b for row in counts.values() for b in row}

    def _probability(self, previous: str, symbol: str) -> float:
        previous = previous if previous in self.counts else UNK
        symbol = symbol if symbol in self._targets else UNK
        vocabulary = len(self._targets) + 1
        row = self.counts.get(previous, {})
        return (row.get(symbol, 0) + self.smoothing) / (self._totals.get(previous, 0) + self.smoothing * vocabulary)

    def _transitions(self, window: Window) -> List[tuple]:
        previous, out = START, []
        for m in window.messages:
            symbol = message_symbol(m)
            out.append((previous, symbol, self._probability(previous, symbol)))
            previous = symbol
        return out

    def score(self, window: Window) -> float:
        steps = self._transitions(window)
        if not steps:
            return 0.0
        return sum(-math.log(p) for _a, _b, p in steps) / len(steps)

    def explain(self, window: Window) -> List[str]:
        rare = Counter()
        for a, b, p in self._transitions(window):
            if p < 0.01:
                rare[(a, b, p)] += 1
        worst = sorted(rare.items(), key=lambda kv: (kv[0][2], -kv[1]))[:3]
        return [f"rare_transition {a}->{b} x{n} (p={p:.1e})" for (a, b, p), n in worst]

    def to_dict(self) -> Dict[str, Any]:
        return {"proto": self.proto, "smoothing": self.smoothing, "counts": self.counts}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarkovDetector":
        try:
            det = cls(smoothing=float(data["smoothing"]))
            det.proto = str(data["proto"])
            counts = {
                str(a): {str(b): int(n) for b, n in row.items()} for a, row in data["counts"].items()
            }
            if any(n < 0 for row in counts.values() for n in row.values()):
                raise ValueError("negative count")
            det._load(counts)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ModelError(f"invalid markov detector state: {exc}") from exc
        return det
