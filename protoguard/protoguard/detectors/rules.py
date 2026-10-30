"""Specification and allow-list rules learned from normal traffic.

Industrial and IoT networks are usually static: the same few clients send the
same few message types to the same devices. This detector learns that set from
normal traffic and flags anything outside it, plus any message that breaks the
protocol specification. It is deliberately strict: every violation is an alert,
so it has no tunable threshold (``fixed_threshold``). Its weakness is the
mirror image of its strength: a legitimate but new behaviour will alert until
the model is retrained.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from ..errors import ModelError
from ..messages import REQUEST, Message
from ..parsers.modbus import READ_FUNCTIONS, WRITE_FUNCTIONS
from ..windows import Window
from .base import Detector


class RuleDetector(Detector):
    name = "rules"
    fixed_threshold = 0.5  # one violation (score >= 1) is enough to alert
    TOPIC_CAP = 2000
    CLIENT_ID_CAP = 500

    def __init__(self, check_clients: bool = False) -> None:
        self.check_clients = check_clients
        self.proto: Optional[str] = None
        self.clients: Set[str] = set()
        # Modbus
        self.functions: Set[int] = set()
        self.units: Set[int] = set()
        self.write_addresses: Set[int] = set()
        self.read_ranges: Dict[int, List[int]] = {}
        # MQTT
        self.topics: Set[str] = set()
        self.topics_overflow = False
        self.topic_roots: Set[str] = set()
        self.filters: Set[str] = set()
        self.client_ids: Set[str] = set()
        self.client_ids_overflow = False

    # ------------------------------------------------------------------ fit
    def fit(self, windows: Sequence[Window]) -> None:
        if not windows:
            raise ModelError("rules: no training windows")
        self.proto = windows[0].proto
        for window in windows:
            for m in window.messages:
                self._learn(m)

    def _learn(self, m: Message) -> None:
        if not m.valid:
            return  # never learn protocol violations as "normal"
        self.clients.add(m.client)
        if m.direction != REQUEST:
            return
        if m.proto == "modbus" and m.function is not None:
            self.functions.add(m.function)
            if m.unit_id is not None:
                self.units.add(m.unit_id)
            if m.address is None:
                return
            if m.function in WRITE_FUNCTIONS:
                self.write_addresses.add(m.address)
            elif m.function in READ_FUNCTIONS:
                last = m.address + (m.quantity or 1) - 1
                span = self.read_ranges.setdefault(m.function, [m.address, last])
                span[0], span[1] = min(span[0], m.address), max(span[1], last)
        elif m.proto == "mqtt":
            if m.kind == "PUBLISH" and m.topic is not None:
                self.topic_roots.add(m.topic.split("/", 1)[0])
                if len(self.topics) < self.TOPIC_CAP:
                    self.topics.add(m.topic)
                elif m.topic not in self.topics:
                    self.topics_overflow = True
            elif m.kind == "SUBSCRIBE":
                self.filters.update(m.filters)
            elif m.kind == "CONNECT" and m.client_id is not None:
                if len(self.client_ids) < self.CLIENT_ID_CAP:
                    self.client_ids.add(m.client_id)
                elif m.client_id not in self.client_ids:
                    self.client_ids_overflow = True

    # ---------------------------------------------------------------- score
    def violations(self, window: Window) -> List[str]:
        found: List[str] = []
        seen: Set[str] = set()

        def add(text: str) -> None:
            if text not in seen:
                seen.add(text)
                found.append(text)

        for m in window.messages:
            if not m.valid:
                for problem in m.problems:
                    add(f"invalid_message:{problem}")
                continue
            if self.check_clients and m.client not in self.clients:
                add(f"unseen_client:{m.client}")
            if m.direction != REQUEST:
                continue
            if window.proto == "modbus":
                self._check_modbus(m, add)
            else:
                self._check_mqtt(m, add)
        return found

    def _check_modbus(self, m: Message, add) -> None:
        if m.function is None:
            return
        if m.function not in self.functions:
            add(f"unseen_function:{m.function}")
        if m.unit_id is not None and m.unit_id not in self.units:
            add(f"unseen_unit:{m.unit_id}")
        if m.address is None:
            return
        if m.function in WRITE_FUNCTIONS:
            if m.address not in self.write_addresses:
                add(f"unseen_write_address:{m.address}")
        elif m.function in READ_FUNCTIONS:
            span = self.read_ranges.get(m.function)
            last = m.address + (m.quantity or 1) - 1
            if span is not None and (m.address < span[0] or last > span[1]):
                add(f"read_out_of_range:fc{m.function}@{m.address}+{m.quantity}")

    def _check_mqtt(self, m: Message, add) -> None:
        if m.kind == "PUBLISH" and m.topic is not None:
            if self.topics_overflow:
                if m.topic.split("/", 1)[0] not in self.topic_roots:
                    add(f"unseen_topic_root:{m.topic.split('/', 1)[0]}")
            elif m.topic not in self.topics:
                add(f"unseen_topic:{m.topic}")
        elif m.kind == "SUBSCRIBE":
            for f in m.filters:
                if f not in self.filters:
                    add(f"unseen_filter:{f}")
        elif m.kind == "CONNECT" and m.client_id is not None:
            if not self.client_ids_overflow and m.client_id not in self.client_ids:
                add(f"unseen_client_id:{m.client_id}")

    def score(self, window: Window) -> float:
        return float(len(self.violations(window)))

    def explain(self, window: Window) -> List[str]:
        return self.violations(window)[:5]

    # -------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proto": self.proto,
            "check_clients": self.check_clients,
            "clients": sorted(self.clients),
            "functions": sorted(self.functions),
            "units": sorted(self.units),
            "write_addresses": sorted(self.write_addresses),
            "read_ranges": {str(k): v for k, v in self.read_ranges.items()},
            "topics": sorted(self.topics),
            "topics_overflow": self.topics_overflow,
            "topic_roots": sorted(self.topic_roots),
            "filters": sorted(self.filters),
            "client_ids": sorted(self.client_ids),
            "client_ids_overflow": self.client_ids_overflow,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleDetector":
        try:
            det = cls(check_clients=bool(data["check_clients"]))
            det.proto = data["proto"]
            det.clients = {str(x) for x in data["clients"]}
            det.functions = {int(x) for x in data["functions"]}
            det.units = {int(x) for x in data["units"]}
            det.write_addresses = {int(x) for x in data["write_addresses"]}
            det.read_ranges = {int(k): [int(v[0]), int(v[1])] for k, v in data["read_ranges"].items()}
            det.topics = {str(x) for x in data["topics"]}
            det.topics_overflow = bool(data["topics_overflow"])
            det.topic_roots = {str(x) for x in data["topic_roots"]}
            det.filters = {str(x) for x in data["filters"]}
            det.client_ids = {str(x) for x in data["client_ids"]}
            det.client_ids_overflow = bool(data["client_ids_overflow"])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ModelError(f"invalid rules detector state: {exc}") from exc
        return det
