"""GuardModel: per-protocol ensembles of detectors with calibrated thresholds.

Training uses *normal* traffic only. Windows are split chronologically: the
earlier part trains the detectors and the later part, which they have never
seen, calibrates the alert thresholds. Each statistical detector's threshold is
set so that at most ``target_fpr / n_detectors`` of the calibration windows
exceed it; a window alerts if any detector exceeds its threshold, so the
union stays near ``target_fpr`` on normal traffic that resembles the
calibration data. Real traffic drifts, so treat the figure as a target, not a
guarantee, and measure the false-positive rate on your own held-out captures.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .detectors import REGISTRY, Detector, IsolationForestDetector, RuleDetector
from .errors import ModelError
from .util import median, threshold_for_fpr
from .windows import Window

SCHEMA_VERSION = 1
DEFAULT_DETECTORS = ("rules", "zscore", "markov", "iforest")
MIN_TRAIN_WINDOWS = 40
MIN_CALIB_WINDOWS = 20


@dataclass
class WindowScore:
    """Scores of one window under every detector of its protocol."""

    window: Window
    raw: Dict[str, float]
    normalized: Dict[str, float]  # (raw - base) / (threshold - base); > 1 means over threshold
    combined: float  # max normalized score, used for ranking and AUC
    triggered: List[str]  # detectors whose raw score exceeded their threshold

    @property
    def alert(self) -> bool:
        return bool(self.triggered)


@dataclass
class Alert:
    proto: str
    client: str
    server: str
    start: float
    end: float
    score: float
    detectors: List[str]
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proto": self.proto, "client": self.client, "server": self.server,
            "start": self.start, "end": self.end, "score": round(self.score, 3),
            "detectors": self.detectors, "reasons": self.reasons,
        }


@dataclass
class ProtocolModel:
    proto: str
    detectors: Dict[str, Detector]
    thresholds: Dict[str, float]
    bases: Dict[str, float]
    train_windows: int = 0
    calibration_windows: int = 0

    def score(self, window: Window) -> WindowScore:
        raw: Dict[str, float] = {}
        normalized: Dict[str, float] = {}
        triggered: List[str] = []
        for name, detector in self.detectors.items():
            value = detector.score(window)
            threshold, base = self.thresholds[name], self.bases[name]
            raw[name] = value
            normalized[name] = (value - base) / max(threshold - base, 1e-9)
            if value > threshold:
                triggered.append(name)
        return WindowScore(window, raw, normalized, max(normalized.values()), triggered)


class GuardModel:
    """Train on normal traffic, then score windows or raise alerts."""

    def __init__(
        self,
        window_s: float = 5.0,
        target_fpr: float = 0.01,
        detectors: Sequence[str] = DEFAULT_DETECTORS,
        seed: int = 0,
        check_clients: bool = False,
        factories: Optional[Mapping[str, Callable[[], Detector]]] = None,
    ) -> None:
        if window_s <= 0:
            raise ModelError("window_s must be positive")
        if not 0.0 < target_fpr < 1.0:
            raise ModelError("target_fpr must be between 0 and 1")
        self.factories = dict(factories or {})
        unknown = [n for n in detectors if n not in REGISTRY and n not in self.factories]
        if unknown or not detectors:
            raise ModelError(f"unknown or missing detectors: {unknown or 'none given'}")
        self.window_s = window_s
        self.target_fpr = target_fpr
        self.detector_names = tuple(detectors)
        self.seed = seed
        self.check_clients = check_clients
        self.protocols: Dict[str, ProtocolModel] = {}
        self.notes: List[str] = []

    # ------------------------------------------------------------------ fit
    def _make(self, name: str) -> Detector:
        if name in self.factories:
            return self.factories[name]()
        if name == "rules":
            return RuleDetector(check_clients=self.check_clients)
        if name == "iforest":
            return IsolationForestDetector(seed=self.seed)
        return REGISTRY[name]()

    def fit(self, windows: Sequence[Window], calibration_fraction: float = 0.3) -> "GuardModel":
        """Train on windows of normal traffic."""
        if not 0.05 <= calibration_fraction <= 0.5:
            raise ModelError("calibration_fraction must be between 0.05 and 0.5")
        self.protocols, self.notes = {}, []
        by_proto: Dict[str, List[Window]] = {}
        for w in windows:
            by_proto.setdefault(w.proto, []).append(w)

        for proto, group in sorted(by_proto.items()):
            group.sort(key=lambda w: w.start)
            cut = group[int(len(group) * (1 - calibration_fraction))].start
            train = [w for w in group if w.start < cut]
            calib = [w for w in group if w.start >= cut]
            if len(train) < MIN_TRAIN_WINDOWS or len(calib) < MIN_CALIB_WINDOWS:
                self.notes.append(
                    f"{proto}: skipped, only {len(train)} training and {len(calib)} calibration windows "
                    f"(need {MIN_TRAIN_WINDOWS}/{MIN_CALIB_WINDOWS}); use a longer capture"
                )
                continue
            self.protocols[proto] = self._fit_protocol(proto, train, calib)
        if not self.protocols:
            raise ModelError("no protocol had enough normal traffic to train on: " + "; ".join(self.notes or ["no windows"]))
        return self

    def _fit_protocol(self, proto: str, train: List[Window], calib: List[Window]) -> ProtocolModel:
        detectors = {name: self._make(name) for name in self.detector_names}
        for detector in detectors.values():
            detector.fit(train)
        calibrated = [n for n, d in detectors.items() if d.fixed_threshold is None]
        alpha = self.target_fpr / max(1, len(calibrated))
        thresholds: Dict[str, float] = {}
        bases: Dict[str, float] = {}
        for name, detector in detectors.items():
            scores = [detector.score(w) for w in calib]
            bases[name] = median(scores)
            if detector.fixed_threshold is not None:
                thresholds[name] = detector.fixed_threshold
            else:
                thresholds[name] = max(detector.min_threshold, threshold_for_fpr(scores, alpha))
        return ProtocolModel(proto, detectors, thresholds, bases, len(train), len(calib))

    # -------------------------------------------------------------- scoring
    def score_windows(self, windows: Sequence[Window]) -> List[WindowScore]:
        """Score every window whose protocol has a trained model (others are skipped)."""
        if not self.protocols:
            raise ModelError("model is not trained")
        return [self.protocols[w.proto].score(w) for w in windows if w.proto in self.protocols]

    def detect(self, windows: Sequence[Window]) -> List[Alert]:
        alerts = []
        for ws in self.score_windows(windows):
            if not ws.alert:
                continue
            pm = self.protocols[ws.window.proto]
            reasons: List[str] = []
            for name in ws.triggered:
                found = pm.detectors[name].explain(ws.window)
                if found:
                    reasons.extend(f"{name}: {r}" for r in found)
                else:
                    reasons.append(f"{name}: score {ws.raw[name]:.3g} above threshold {pm.thresholds[name]:.3g}")
            w = ws.window
            alerts.append(Alert(w.proto, w.client, w.server, w.start, w.end, ws.combined, ws.triggered, reasons))
        return alerts

    # --------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        if self.factories:
            raise ModelError("models with custom detector factories cannot be saved")
        return {
            "schema_version": SCHEMA_VERSION,
            "window_s": self.window_s,
            "target_fpr": self.target_fpr,
            "detectors": list(self.detector_names),
            "seed": self.seed,
            "check_clients": self.check_clients,
            "notes": self.notes,
            "protocols": {
                proto: {
                    "train_windows": pm.train_windows,
                    "calibration_windows": pm.calibration_windows,
                    "thresholds": pm.thresholds,
                    "bases": pm.bases,
                    "detectors": {name: d.to_dict() for name, d in pm.detectors.items()},
                }
                for proto, pm in self.protocols.items()
            },
        }

    def save(self, path) -> None:
        """Write the model as JSON (atomically; never pickle)."""
        payload = json.dumps(self.to_dict(), indent=1)
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path) -> "GuardModel":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ModelError(f"cannot read model {path}: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Any) -> "GuardModel":
        try:
            if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
                raise ModelError(f"unsupported model format (expected schema_version {SCHEMA_VERSION})")
            model = cls(
                window_s=float(data["window_s"]), target_fpr=float(data["target_fpr"]),
                detectors=tuple(data["detectors"]), seed=int(data["seed"]),
                check_clients=bool(data["check_clients"]),
            )
            model.notes = [str(n) for n in data.get("notes", [])]
            for proto, entry in data["protocols"].items():
                detectors = {}
                for name, state in entry["detectors"].items():
                    if name not in REGISTRY:
                        raise ModelError(f"unknown detector '{name}' in model file")
                    detectors[name] = REGISTRY[name].from_dict(state)
                thresholds = {str(k): float(v) for k, v in entry["thresholds"].items()}
                bases = {str(k): float(v) for k, v in entry["bases"].items()}
                if set(thresholds) != set(detectors) or set(bases) != set(detectors):
                    raise ModelError("thresholds do not match detectors")
                if not all(math.isfinite(v) for v in list(thresholds.values()) + list(bases.values())):
                    raise ModelError("non-finite threshold in model file")
                model.protocols[str(proto)] = ProtocolModel(
                    str(proto), detectors, thresholds, bases,
                    int(entry["train_windows"]), int(entry["calibration_windows"]),
                )
        except ModelError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ModelError(f"corrupt model file: {exc}") from exc
        return model
