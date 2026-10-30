"""Evaluate a trained model on a labelled capture.

Metrics are computed per *window* (a window is positive if it overlaps a
labelled attack interval, see ``labels.py``). Besides overall precision,
recall, F1, false-positive rate and AUC, the report includes recall per attack
type, the delay until each attack is first alerted, and an ablation that scores
every detector alone, so you can see which components actually help.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence

from .labels import Interval, label_windows
from .metrics import confusion_metrics, roc_auc
from .model import GuardModel, WindowScore
from .windows import Window


def _summary(scores: Sequence[WindowScore], alerted: Sequence[bool], values: Sequence[float]) -> Dict[str, Any]:
    y_true = [s.window.label for s in scores]
    result = confusion_metrics(y_true, [int(a) for a in alerted])
    result["auc"] = roc_auc(y_true, values)
    result["windows"] = len(scores)
    result["attack_windows"] = sum(y_true)
    return result


def evaluate_model(model: GuardModel, windows: Sequence[Window], intervals: Sequence[Interval]) -> Dict[str, Any]:
    """Label ``windows`` with ``intervals``, score them, and report."""
    index = label_windows(windows, intervals)
    scored = model.score_windows(windows)
    if not scored:
        raise ValueError("no windows of a trained protocol in the test capture")

    result: Dict[str, Any] = {
        "overall": _summary(scored, [s.alert for s in scored], [s.combined for s in scored]),
        "by_protocol": {},
        "by_attack": {},
        "detectors": {},
        "intervals": {},
    }

    for proto in sorted({s.window.proto for s in scored}):
        part = [s for s in scored if s.window.proto == proto]
        result["by_protocol"][proto] = _summary(part, [s.alert for s in part], [s.combined for s in part])

    by_attack: Dict[str, List[WindowScore]] = {}
    for s in scored:
        if s.window.label:
            by_attack.setdefault(s.window.attack or "attack", []).append(s)
    for attack, part in sorted(by_attack.items()):
        hits = sum(1 for s in part if s.alert)
        result["by_attack"][attack] = {"windows": len(part), "alerted": hits, "recall": hits / len(part)}

    names = sorted({n for s in scored for n in s.raw})
    for name in names:
        part = [s for s in scored if name in s.raw]
        result["detectors"][name] = _summary(
            part, [name in s.triggered for s in part], [s.normalized[name] for s in part]
        )

    # Per-interval detection and delay (time from attack start to the end of the first alerted window).
    per_interval: Dict[Interval, List[WindowScore]] = {iv: [] for iv in intervals}
    for s in scored:
        for iv in index.matches(s.window):
            per_interval[iv].append(s)
    delays, detected = [], 0
    for iv, part in per_interval.items():
        alerted = [s for s in part if s.alert]
        if alerted:
            detected += 1
            delays.append(max(0.0, min(s.window.end for s in alerted) - iv.start))
    result["intervals"] = {
        "count": len(intervals),
        "detected": detected,
        "detection_rate": detected / len(intervals) if intervals else None,
        "mean_delay_s": statistics.fmean(delays) if delays else None,
        "median_delay_s": statistics.median(delays) if delays else None,
    }
    return result


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_report(result: Dict[str, Any]) -> str:
    """Plain-text report for the terminal."""
    lines = []
    o = result["overall"]
    lines.append(f"windows: {o['windows']} ({o['attack_windows']} overlap an attack)")
    lines.append(
        f"overall: precision {o['precision']:.3f}  recall {o['recall']:.3f}  f1 {o['f1']:.3f}  "
        f"FPR {o['false_positive_rate']:.3f}  AUC {_fmt(o['auc'])}"
    )
    lines.append(f"         tp={o['tp']} fp={o['fp']} tn={o['tn']} fn={o['fn']}")
    lines.append("")
    lines.append("by protocol:")
    for proto, m in result["by_protocol"].items():
        lines.append(
            f"  {proto:7s} precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
            f"FPR {m['false_positive_rate']:.3f}  AUC {_fmt(m['auc'])}  ({m['windows']} windows)"
        )
    lines.append("")
    lines.append("recall by attack (share of attack windows alerted):")
    for attack, m in result["by_attack"].items():
        lines.append(f"  {attack:28s} {m['recall']:.2f}  ({m['alerted']}/{m['windows']})")
    iv = result["intervals"]
    lines.append("")
    lines.append(
        f"attacks detected at least once: {iv['detected']}/{iv['count']}  "
        f"(mean delay {_fmt(iv['mean_delay_s'], 1)} s, median {_fmt(iv['median_delay_s'], 1)} s)"
    )
    lines.append("")
    lines.append("each detector alone:")
    for name, m in result["detectors"].items():
        lines.append(
            f"  {name:10s} precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
            f"FPR {m['false_positive_rate']:.3f}  AUC {_fmt(m['auc'])}"
        )
    return "\n".join(lines)
