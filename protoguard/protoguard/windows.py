"""Group messages into fixed time windows per channel and compute features.

A *channel* is ``(protocol, client IP, server IP)``. Windows are tumbling and
aligned to multiples of ``window_s`` on the capture clock, so a training
capture and a test capture produce comparable windows.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .messages import REQUEST, RESPONSE, Message
from .parsers.modbus import READ_FUNCTIONS, WRITE_FUNCTIONS
from .util import mean_std

MODBUS_FEATURES = [
    "msg_rate", "req_rate", "resp_ratio", "read_ratio", "write_ratio", "other_ratio",
    "n_functions", "n_units", "n_addresses", "exception_ratio", "invalid_ratio",
    "mean_quantity", "max_quantity", "iat_mean", "iat_std", "mean_size", "max_size",
]
MQTT_FEATURES = [
    "msg_rate", "pub_rate", "deliver_rate", "sub_rate", "connect_rate", "other_ratio",
    "n_topics", "n_client_ids", "wildcard_subs", "n_filters",
    "mean_payload_len", "max_payload_len", "payload_len_std", "mean_entropy",
    "mean_qos", "retain_ratio", "invalid_ratio", "iat_mean", "iat_std", "mean_size", "max_size",
]
FEATURES: Dict[str, List[str]] = {"modbus": MODBUS_FEATURES, "mqtt": MQTT_FEATURES}


@dataclass
class Window:
    """Messages of one channel within one time window, with derived features."""

    proto: str
    client: str
    server: str
    start: float
    end: float
    messages: List[Message]
    features: Dict[str, float]
    label: int = 0  # 1 = overlaps a labelled attack (set by labels.label_windows)
    attack: Optional[str] = None

    @property
    def channel(self) -> Tuple[str, str, str]:
        return (self.proto, self.client, self.server)


def vector(window: Window) -> List[float]:
    """Feature vector in the protocol's fixed order, log1p-transformed.

    The transform tames the heavy tails of counts and rates so that one flood
    does not dominate every distance or z-score.
    """
    return [math.log1p(max(0.0, window.features[name])) for name in FEATURES[window.proto]]


def _iat(times: Sequence[float], window_s: float) -> Tuple[float, float]:
    if len(times) < 2:
        return window_s, 0.0
    ordered = sorted(times)
    return mean_std([b - a for a, b in zip(ordered, ordered[1:])])


def _modbus_features(msgs: Sequence[Message], window_s: float) -> Dict[str, float]:
    reqs = [m for m in msgs if m.direction == REQUEST]
    resps = [m for m in msgs if m.direction == RESPONSE]
    n_req = len(reqs)
    reads = sum(1 for m in reqs if m.function in READ_FUNCTIONS)
    writes = sum(1 for m in reqs if m.function in WRITE_FUNCTIONS)
    quantities = [m.quantity for m in reqs if m.quantity is not None]
    sizes = [m.size for m in msgs]
    iat_mean, iat_std = _iat([m.ts for m in reqs], window_s)
    return {
        "msg_rate": len(msgs) / window_s,
        "req_rate": n_req / window_s,
        "resp_ratio": len(resps) / max(1, n_req),
        "read_ratio": reads / max(1, n_req),
        "write_ratio": writes / max(1, n_req),
        "other_ratio": (n_req - reads - writes) / max(1, n_req),
        "n_functions": float(len({m.function for m in reqs if m.function is not None})),
        "n_units": float(len({m.unit_id for m in reqs if m.unit_id is not None})),
        "n_addresses": float(len({m.address for m in reqs if m.address is not None})),
        "exception_ratio": sum(1 for m in resps if m.exception is not None) / max(1, len(resps)),
        "invalid_ratio": sum(1 for m in msgs if not m.valid) / max(1, len(msgs)),
        "mean_quantity": sum(quantities) / len(quantities) if quantities else 0.0,
        "max_quantity": float(max(quantities)) if quantities else 0.0,
        "iat_mean": iat_mean,
        "iat_std": iat_std,
        "mean_size": sum(sizes) / max(1, len(sizes)),
        "max_size": float(max(sizes)) if sizes else 0.0,
    }


def _mqtt_features(msgs: Sequence[Message], window_s: float) -> Dict[str, float]:
    pubs_in = [m for m in msgs if m.kind == "PUBLISH" and m.direction == REQUEST]
    pubs_out = [m for m in msgs if m.kind == "PUBLISH" and m.direction == RESPONSE]
    pubs = pubs_in + pubs_out
    subs = [m for m in msgs if m.kind in ("SUBSCRIBE", "UNSUBSCRIBE")]
    connects = [m for m in msgs if m.kind == "CONNECT"]
    filters = [f for m in subs for f in m.filters]
    lengths = [float(m.payload_len) for m in pubs if m.payload_len is not None]
    entropies = [m.payload_entropy for m in pubs if m.payload_entropy is not None]
    qos = [m.qos for m in pubs if m.qos is not None]
    sizes = [m.size for m in msgs]
    iat_mean, iat_std = _iat([m.ts for m in msgs if m.direction == REQUEST], window_s)
    n = max(1, len(msgs))
    named = len(pubs) + len(subs) + len(connects)
    return {
        "msg_rate": len(msgs) / window_s,
        "pub_rate": len(pubs_in) / window_s,
        "deliver_rate": len(pubs_out) / window_s,
        "sub_rate": len(subs) / window_s,
        "connect_rate": len(connects) / window_s,
        "other_ratio": (len(msgs) - named) / n,
        "n_topics": float(len({m.topic for m in pubs_in if m.topic is not None})),
        "n_client_ids": float(len({m.client_id for m in connects if m.client_id is not None})),
        "wildcard_subs": float(sum(1 for f in filters if "#" in f or "+" in f)),
        "n_filters": float(len(filters)),
        "mean_payload_len": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_payload_len": max(lengths) if lengths else 0.0,
        "payload_len_std": mean_std(lengths)[1],
        "mean_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        "mean_qos": sum(qos) / len(qos) if qos else 0.0,
        "retain_ratio": sum(1 for m in pubs if m.retain) / max(1, len(pubs)),
        "invalid_ratio": sum(1 for m in msgs if not m.valid) / n,
        "iat_mean": iat_mean,
        "iat_std": iat_std,
        "mean_size": sum(sizes) / n,
        "max_size": float(max(sizes)) if sizes else 0.0,
    }


def extract_features(proto: str, msgs: Sequence[Message], window_s: float) -> Dict[str, float]:
    if proto == "modbus":
        return _modbus_features(msgs, window_s)
    if proto == "mqtt":
        return _mqtt_features(msgs, window_s)
    raise ValueError(f"unsupported protocol: {proto}")


def build_windows(messages: Iterable[Message], window_s: float = 5.0) -> List[Window]:
    """Group messages into windows and compute their features.

    Returns windows sorted by start time. Everything is held in memory, so very
    large captures should be split first.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    groups: Dict[Tuple[str, str, str, int], List[Message]] = defaultdict(list)
    for m in messages:
        groups[(m.proto, m.client, m.server, int(math.floor(m.ts / window_s)))].append(m)

    windows: List[Window] = []
    for (proto, client, server, index), msgs in groups.items():
        msgs.sort(key=lambda m: m.ts)
        windows.append(Window(
            proto=proto, client=client, server=server,
            start=index * window_s, end=(index + 1) * window_s,
            messages=msgs, features=extract_features(proto, msgs, window_s),
        ))
    windows.sort(key=lambda w: (w.start, w.proto, w.client, w.server))
    return windows
