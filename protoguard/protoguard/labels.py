"""Ground-truth labels: attack time intervals, and applying them to windows.

Label files are CSV with a header. Required columns: ``start`` and ``end``
(seconds on the same clock as the capture, i.e. Unix time). Optional columns:
``attack`` (name, default "attack") and ``src_ip`` (if set, only windows whose
client IP matches are labelled, which separates an attacker from bystanders on
the same network) and ``dst_ip`` (if set, the window's server IP must match too, so
an insider's normal traffic to other devices is not labelled as attack).
"""
from __future__ import annotations

import bisect
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .errors import ProtoGuardError
from .windows import Window


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    attack: str = "attack"
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None


def load_labels(path) -> List[Interval]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or not {"start", "end"} <= set(reader.fieldnames):
                raise ProtoGuardError("label file needs 'start' and 'end' columns")
            intervals = []
            for line, row in enumerate(reader, start=2):
                try:
                    start, end = float(row["start"]), float(row["end"])
                except (TypeError, ValueError) as exc:
                    raise ProtoGuardError(f"{path}:{line}: start/end must be numbers") from exc
                if not end > start:
                    raise ProtoGuardError(f"{path}:{line}: end must be greater than start")
                intervals.append(Interval(
                    start, end, (row.get("attack") or "attack").strip(),
                    (row.get("src_ip") or "").strip() or None, (row.get("dst_ip") or "").strip() or None,
                ))
    except OSError as exc:
        raise ProtoGuardError(f"cannot read label file {path}: {exc}") from exc
    return intervals


def save_labels(path, intervals: Iterable[Interval]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["start", "end", "attack", "src_ip", "dst_ip"])
        for iv in sorted(intervals, key=lambda i: i.start):
            writer.writerow([f"{iv.start:.3f}", f"{iv.end:.3f}", iv.attack, iv.src_ip or "", iv.dst_ip or ""])


class IntervalIndex:
    """Find the intervals that overlap a window without scanning all of them."""

    def __init__(self, intervals: Sequence[Interval]) -> None:
        groups: Dict[Optional[str], List[Interval]] = defaultdict(list)
        for iv in intervals:
            groups[iv.src_ip].append(iv)
        self._groups = {ip: sorted(ivs, key=lambda i: i.start) for ip, ivs in groups.items()}
        self._starts = {ip: [i.start for i in ivs] for ip, ivs in self._groups.items()}
        self._longest = {ip: max(i.end - i.start for i in ivs) for ip, ivs in self._groups.items()}

    def matches(self, window: Window) -> List[Interval]:
        found: List[Interval] = []
        for ip in (None, window.client):
            ivs = self._groups.get(ip)
            if not ivs:
                continue
            starts = self._starts[ip]
            hi = bisect.bisect_left(starts, window.end)  # intervals starting before the window ends
            lo = bisect.bisect_left(starts, window.start - self._longest[ip])
            for iv in ivs[lo:hi]:
                if iv.end > window.start and iv.dst_ip in (None, window.server):
                    found.append(iv)
        return found


def label_windows(windows: Sequence[Window], intervals: Sequence[Interval]) -> IntervalIndex:
    """Set ``label`` (1 = attack) and ``attack`` on every window. Returns the index used."""
    index = IntervalIndex(intervals)
    for w in windows:
        hits = index.matches(w)
        w.label = 1 if hits else 0
        w.attack = hits[0].attack if hits else None
    return index
