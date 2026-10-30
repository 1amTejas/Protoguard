"""MQTT 3.1 / 3.1.1 / 5 control-packet parser and packet builders.

Parses plaintext MQTT (typically port 1883). TLS traffic (8883) cannot be
parsed. The protocol level is learned from the CONNECT packet and shared by
both directions of a connection, so MQTT 5 properties are skipped correctly
once a CONNECT has been seen. If a capture starts mid-connection the level
defaults to 3.1.1.
"""
from __future__ import annotations

import struct
from typing import Dict, List, Optional, Sequence, Tuple

from ..messages import REQUEST, RESPONSE, Message
from ..util import shannon_entropy

TYPES = {
    1: "CONNECT", 2: "CONNACK", 3: "PUBLISH", 4: "PUBACK", 5: "PUBREC", 6: "PUBREL", 7: "PUBCOMP",
    8: "SUBSCRIBE", 9: "SUBACK", 10: "UNSUBSCRIBE", 11: "UNSUBACK", 12: "PINGREQ", 13: "PINGRESP",
    14: "DISCONNECT", 15: "AUTH",
}
CLIENT_ONLY = frozenset({1, 8, 10, 12})
SERVER_ONLY = frozenset({2, 9, 11, 13})
REQUIRED_FLAGS = {6: 0x02, 8: 0x02, 10: 0x02}  # everything except PUBLISH otherwise needs 0
MAX_PACKET = 1 << 20  # larger packets are treated as malformed/oversize


class _Reader:
    """Bounds-checked cursor over a packet body. Overruns raise ValueError."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise ValueError("truncated")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack(">H", self.take(2))[0]

    def string(self) -> str:
        return self.take(self.u16()).decode("utf-8", "replace")

    def varint(self) -> int:
        value, mult = 0, 1
        for _ in range(4):
            byte = self.u8()
            value += (byte & 0x7F) * mult
            if not byte & 0x80:
                return value
            mult *= 128
        raise ValueError("bad varint")

    def skip_properties(self) -> None:
        self.take(self.varint())

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def rest(self) -> bytes:
        out = self.data[self.pos :]
        self.pos = len(self.data)
        return out


def _read_varint(buf: bytearray, pos: int) -> Optional[Tuple[int, int]]:
    """Decode the remaining-length field. ``None`` = need more bytes; ValueError = malformed."""
    value, mult = 0, 1
    for i in range(4):
        if pos + i >= len(buf):
            return None
        byte = buf[pos + i]
        value += (byte & 0x7F) * mult
        if not byte & 0x80:
            return value, i + 1
        mult *= 128
    raise ValueError("bad remaining length")


class MqttParser:
    """Parse MQTT control packets from one direction of a TCP stream."""

    proto = "mqtt"

    def __init__(self, direction: str, shared: Optional[Dict[str, int]] = None, max_packet: int = MAX_PACKET) -> None:
        self.direction = direction
        self.shared = shared if shared is not None else {"version": 4}
        self.max_packet = max_packet

    # ------------------------------------------------------------------ API
    def parse(self, buf: bytearray, ts: float, client: str, server: str) -> Tuple[List[Message], int]:
        messages: List[Message] = []
        pos, n = 0, len(buf)
        while n - pos >= 2:
            ptype, flags = buf[pos] >> 4, buf[pos] & 0x0F
            if ptype == 0:
                messages.append(self._bad(ts, client, server, n - pos, "reserved_type"))
                return messages, n
            try:
                header = _read_varint(buf, pos + 1)
            except ValueError:
                messages.append(self._bad(ts, client, server, n - pos, "bad_remaining_length"))
                return messages, n
            if header is None:
                break
            remaining, vlen = header
            if remaining > self.max_packet:
                messages.append(self._bad(ts, client, server, n - pos, "oversize"))
                return messages, n
            total = 1 + vlen + remaining
            if n - pos < total:
                break
            body = bytes(buf[pos + 1 + vlen : pos + total])
            messages.append(self._decode(ptype, flags, body, total, ts, client, server))
            pos += total
        return messages, pos

    # ------------------------------------------------------------- internals
    def _bad(self, ts: float, client: str, server: str, size: int, problem: str) -> Message:
        return Message(
            ts=ts, proto="mqtt", client=client, server=server, direction=self.direction,
            kind="invalid", size=size, valid=False, problems=(problem,),
        )

    def _decode(self, ptype: int, flags: int, body: bytes, size: int, ts: float, client: str, server: str) -> Message:
        problems: List[str] = []
        if ptype in CLIENT_ONLY and self.direction != REQUEST:
            problems.append("wrong_direction")
        if ptype in SERVER_ONLY and self.direction != RESPONSE:
            problems.append("wrong_direction")
        if ptype != 3 and flags != REQUIRED_FLAGS.get(ptype, 0):
            problems.append("bad_flags")

        fields: Dict[str, object] = {}
        try:
            reader = _Reader(body)
            if ptype == 1:
                fields = self._connect(reader, problems)
            elif ptype == 3:
                fields = self._publish(reader, flags, problems)
            elif ptype == 8:
                fields = self._subscribe(reader, True, problems)
            elif ptype == 10:
                fields = self._subscribe(reader, False, problems)
        except ValueError:
            problems.append("truncated_body")

        return Message(
            ts=ts, proto="mqtt", client=client, server=server, direction=self.direction,
            kind=TYPES.get(ptype, f"TYPE{ptype}"), size=size, valid=not problems, problems=tuple(problems),
            topic=fields.get("topic"), qos=fields.get("qos"), retain=bool(fields.get("retain", False)),
            payload_len=fields.get("payload_len"), payload_entropy=fields.get("payload_entropy"),
            client_id=fields.get("client_id"), filters=tuple(fields.get("filters", ())),
        )

    def _connect(self, reader: _Reader, problems: List[str]) -> Dict[str, object]:
        name = reader.string()
        level = reader.u8()
        reader.u8()  # connect flags
        reader.u16()  # keep-alive
        if name not in ("MQTT", "MQIsdp"):
            problems.append("bad_protocol_name")
        if level in (3, 4, 5):
            self.shared["version"] = level
        else:
            problems.append("bad_protocol_level")
        if level == 5:
            reader.skip_properties()
        return {"client_id": reader.string()}

    def _publish(self, reader: _Reader, flags: int, problems: List[str]) -> Dict[str, object]:
        topic = reader.string()
        qos = (flags >> 1) & 0x03
        if qos == 3:
            problems.append("bad_qos")
        if qos > 0:
            reader.u16()  # packet identifier
        version = self.shared["version"]
        if version == 5:
            reader.skip_properties()
        payload = reader.rest()
        if "+" in topic or "#" in topic:
            problems.append("wildcard_in_publish")
        if not topic and version < 5:
            problems.append("empty_topic")
        return {
            "topic": topic, "qos": qos, "retain": bool(flags & 1),
            "payload_len": len(payload), "payload_entropy": shannon_entropy(payload),
        }

    def _subscribe(self, reader: _Reader, with_options: bool, problems: List[str]) -> Dict[str, object]:
        reader.u16()  # packet identifier
        if self.shared["version"] == 5:
            reader.skip_properties()
        filters: List[str] = []
        while reader.remaining() > 0:
            filters.append(reader.string())
            if with_options:
                reader.u8()
        if not filters:
            problems.append("empty_subscribe")
        return {"filters": tuple(filters)}


# --------------------------------------------------------------------------
# Packet builders (used by the simulator and the tests)
# --------------------------------------------------------------------------
def encode_varint(value: int) -> bytes:
    if not 0 <= value <= 268_435_455:
        raise ValueError("remaining length out of range")
    out = bytearray()
    while True:
        byte, value = value % 128, value // 128
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def packet(ptype: int, flags: int, body: bytes = b"") -> bytes:
    return bytes([(ptype << 4) | (flags & 0x0F)]) + encode_varint(len(body)) + body


def encode_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack(">H", len(raw)) + raw


def connect(client_id: str, keepalive: int = 60, level: int = 4, clean: bool = True) -> bytes:
    body = encode_string("MQTT") + bytes([level, 0x02 if clean else 0x00]) + struct.pack(">H", keepalive)
    if level == 5:
        body += b"\x00"  # empty properties
    return packet(1, 0, body + encode_string(client_id))


def connack(code: int = 0) -> bytes:
    return packet(2, 0, bytes([0, code]))


def publish(topic: str, payload: bytes, qos: int = 0, retain: bool = False, packet_id: int = 1) -> bytes:
    body = encode_string(topic) + (struct.pack(">H", packet_id) if qos else b"") + payload
    return packet(3, (qos << 1) | int(retain), body)


def puback(packet_id: int) -> bytes:
    return packet(4, 0, struct.pack(">H", packet_id))


def subscribe(packet_id: int, filters: Sequence[Tuple[str, int]]) -> bytes:
    body = struct.pack(">H", packet_id) + b"".join(encode_string(f) + bytes([qos]) for f, qos in filters)
    return packet(8, 2, body)


def suback(packet_id: int, codes: Sequence[int]) -> bytes:
    return packet(9, 0, struct.pack(">H", packet_id) + bytes(codes))


def pingreq() -> bytes:
    return packet(12, 0)


def pingresp() -> bytes:
    return packet(13, 0)


def disconnect() -> bytes:
    return packet(14, 0)
