"""Read packet captures and turn them into application messages.

Pipeline: capture file (pcap / pcapng, optionally .gz) -> frames -> TCP
segments -> per-direction stream reassembly -> Modbus/MQTT messages.

Limits worth knowing: IPv4 and IPv6 over Ethernet, Linux "cooked" and raw-IP
link types only; IP fragments and non-TCP packets are skipped; TLS-encrypted
traffic cannot be parsed. Reassembly is deliberately simple: in-order data and
retransmissions are handled, and a sequence gap resets the stream.
"""
from __future__ import annotations

import gzip
import socket
import struct
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .errors import CaptureError
from .messages import REQUEST, RESPONSE, Message
from .parsers.modbus import ModbusParser
from .parsers.mqtt import MAX_PACKET, MqttParser

SEQ_MASK = 0xFFFFFFFF
MAX_FRAME = 16 * 1024 * 1024
MAX_BLOCK = 64 * 1024 * 1024

_PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),
}
_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"


@dataclass
class Segment:
    """One TCP segment with its payload."""

    ts: float
    src: str
    sport: int
    dst: str
    dport: int
    seq: int
    flags: int
    payload: bytes


# --------------------------------------------------------------------------
# Reading capture files
# --------------------------------------------------------------------------
def _open(path) -> BinaryIO:
    try:
        if str(path).endswith(".gz"):
            return gzip.open(path, "rb")  # type: ignore[return-value]
        return open(path, "rb")
    except OSError as exc:
        raise CaptureError(f"cannot open capture {path}: {exc}") from exc


def iter_frames(path) -> Iterator[Tuple[float, int, bytes]]:
    """Yield ``(timestamp, linktype, frame_bytes)`` from a pcap or pcapng file."""
    with _open(path) as fh:
        magic = fh.read(4)
        if magic in _PCAP_MAGICS:
            yield from _iter_pcap(fh, magic)
        elif magic == _PCAPNG_MAGIC:
            yield from _iter_pcapng(fh, magic)
        else:
            raise CaptureError(f"{path} is not a pcap or pcapng file")


def _iter_pcap(fh: BinaryIO, magic: bytes) -> Iterator[Tuple[float, int, bytes]]:
    endian, resolution = _PCAP_MAGICS[magic]
    rest = fh.read(20)
    if len(rest) < 20:
        raise CaptureError("truncated pcap header")
    linktype = struct.unpack(endian + "HHiIII", rest)[5]
    record = struct.Struct(endian + "IIII")
    while True:
        head = fh.read(16)
        if len(head) < 16:
            return
        sec, frac, caplen, _orig = record.unpack(head)
        if caplen > MAX_FRAME:
            raise CaptureError("corrupt pcap: oversized record")
        data = fh.read(caplen)
        if len(data) < caplen:
            return  # truncated final record
        yield sec + frac * resolution, linktype, data


def _iter_pcapng(fh: BinaryIO, first: bytes) -> Iterator[Tuple[float, int, bytes]]:
    endian = "<"
    interfaces: List[Tuple[int, float]] = []
    pending: Optional[bytes] = first
    while True:
        head = (pending or b"") + fh.read(8 - len(pending or b""))
        pending = None
        if len(head) < 8:
            return
        if head[:4] == _PCAPNG_MAGIC:
            order = fh.read(4)
            if order == b"\x4d\x3c\x2b\x1a":
                endian = "<"
            elif order == b"\x1a\x2b\x3c\x4d":
                endian = ">"
            else:
                raise CaptureError("bad pcapng byte-order magic")
            total = struct.unpack(endian + "I", head[4:8])[0]
            if total < 12 or total > MAX_BLOCK:
                raise CaptureError("corrupt pcapng section header")
            fh.read(total - 12)
            interfaces = []
            continue
        btype, total = struct.unpack(endian + "II", head)
        if total < 12 or total > MAX_BLOCK:
            raise CaptureError("corrupt pcapng block")
        body = fh.read(total - 8)
        if len(body) < total - 8:
            return
        body = body[:-4]  # drop the trailing block length
        if btype == 1:  # interface description
            linktype = struct.unpack(endian + "H", body[:2])[0]
            interfaces.append((linktype, _pcapng_resolution(body, endian)))
        elif btype == 6:  # enhanced packet
            if len(body) < 20:
                continue
            iface, hi, lo, caplen, _orig = struct.unpack(endian + "IIIII", body[:20])
            if iface >= len(interfaces):
                continue
            linktype, resolution = interfaces[iface]
            data = body[20 : 20 + caplen]
            if len(data) < caplen:
                continue
            yield ((hi << 32) | lo) * resolution, linktype, data


def _pcapng_resolution(body: bytes, endian: str) -> float:
    resolution, off = 1e-6, 8
    while off + 4 <= len(body):
        code, length = struct.unpack(endian + "HH", body[off : off + 4])
        if code == 0:
            break
        if code == 9 and length >= 1 and off + 4 < len(body):
            value = body[off + 4]
            resolution = 2.0 ** -(value & 0x7F) if value & 0x80 else 10.0 ** -value
        off += 4 + ((length + 3) & ~3)
    return resolution


# --------------------------------------------------------------------------
# Decoding frames
# --------------------------------------------------------------------------
def decode_frame(ts: float, linktype: int, data: bytes) -> Optional[Segment]:
    """Decode an Ethernet/IP/TCP frame. Returns ``None`` for anything else."""
    off = 0
    if linktype == 1:  # Ethernet
        if len(data) < 14:
            return None
        ethertype = struct.unpack("!H", data[12:14])[0]
        off = 14
        while ethertype in (0x8100, 0x88A8):  # VLAN tags
            if len(data) < off + 4:
                return None
            ethertype = struct.unpack("!H", data[off + 2 : off + 4])[0]
            off += 4
    elif linktype == 113:  # Linux cooked capture
        if len(data) < 16:
            return None
        ethertype = struct.unpack("!H", data[14:16])[0]
        off = 16
    elif linktype in (101, 12, 14):  # raw IP
        version = data[0] >> 4 if data else 0
        ethertype = 0x0800 if version == 4 else 0x86DD if version == 6 else 0
    else:
        return None

    ip = data[off:]
    if ethertype == 0x0800:
        if len(ip) < 20:
            return None
        ihl = (ip[0] & 0x0F) * 4
        total = struct.unpack("!H", ip[2:4])[0]
        if ihl < 20 or total < ihl or len(ip) < ihl:
            return None
        if struct.unpack("!H", ip[6:8])[0] & 0x3FFF:  # fragment
            return None
        if ip[9] != 6:
            return None
        src = ".".join(str(b) for b in ip[12:16])
        dst = ".".join(str(b) for b in ip[16:20])
        tcp = ip[ihl:total]  # bounded by the IP length: drops Ethernet padding
    elif ethertype == 0x86DD:
        if len(ip) < 40 or ip[6] != 6:
            return None
        plen = struct.unpack("!H", ip[4:6])[0]
        src = socket.inet_ntop(socket.AF_INET6, bytes(ip[8:24]))
        dst = socket.inet_ntop(socket.AF_INET6, bytes(ip[24:40]))
        tcp = ip[40 : 40 + plen]
    else:
        return None

    if len(tcp) < 20:
        return None
    sport, dport, seq = struct.unpack("!HHI", tcp[:8])
    doff = (tcp[12] >> 4) * 4
    if doff < 20 or len(tcp) < doff:
        return None
    return Segment(ts, src, sport, dst, dport, seq, tcp[13], bytes(tcp[doff:]))


def read_segments(path, stats: Optional[Counter] = None) -> Iterator[Segment]:
    """Yield TCP segments from a capture file."""
    for ts, linktype, data in iter_frames(path):
        segment = decode_frame(ts, linktype, data)
        if stats is not None:
            stats["frames"] += 1
            if segment is None:
                stats["non_tcp_or_unsupported"] += 1
        if segment is not None:
            yield segment


# --------------------------------------------------------------------------
# TCP stream reassembly
# --------------------------------------------------------------------------
@dataclass
class _Stream:
    parser: object
    proto: str
    direction: str
    client: str
    server: str
    conn_key: tuple
    max_buffer: int
    buf: bytearray = field(default_factory=bytearray)
    next_seq: Optional[int] = None


class StreamReassembler:
    """Turn TCP segments into Modbus/MQTT messages.

    A segment is attributed to a protocol by its port: traffic to or from one
    of ``modbus_ports`` / ``mqtt_ports`` is parsed, everything else is ignored.
    """

    def __init__(
        self,
        modbus_ports: Sequence[int] = (502,),
        mqtt_ports: Sequence[int] = (1883,),
        max_streams: int = 100_000,
    ) -> None:
        self.modbus_ports = set(modbus_ports)
        self.mqtt_ports = set(mqtt_ports)
        if self.modbus_ports & self.mqtt_ports:
            raise ValueError("a port cannot be both Modbus and MQTT")
        self.max_streams = max_streams
        self.streams: "OrderedDict[tuple, _Stream]" = OrderedDict()
        self.connections: Dict[tuple, Dict[str, int]] = {}
        self.stats: Counter = Counter()

    def _classify(self, seg: Segment) -> Optional[Tuple[str, str, str, str, tuple]]:
        for proto, ports in (("modbus", self.modbus_ports), ("mqtt", self.mqtt_ports)):
            if seg.dport in ports:
                return proto, REQUEST, seg.src, seg.dst, (seg.src, seg.sport, seg.dst, seg.dport)
            if seg.sport in ports:
                return proto, RESPONSE, seg.dst, seg.src, (seg.dst, seg.dport, seg.src, seg.sport)
        return None

    def _new_stream(self, proto: str, direction: str, client: str, server: str, conn_key: tuple) -> _Stream:
        if proto == "modbus":
            parser: object = ModbusParser(direction)
            max_buffer = 4096
        else:
            shared = self.connections.setdefault(conn_key, {"version": 4})
            parser = MqttParser(direction, shared)
            max_buffer = MAX_PACKET + 16
        return _Stream(parser, proto, direction, client, server, conn_key, max_buffer)

    def feed(self, seg: Segment) -> List[Message]:
        """Consume one segment; return any messages it completed."""
        info = self._classify(seg)
        if info is None:
            self.stats["ignored_segments"] += 1
            return []
        proto, direction, client, server, conn_key = info
        key = (seg.src, seg.sport, seg.dst, seg.dport)
        stream = self.streams.get(key)
        if stream is None:
            if len(self.streams) >= self.max_streams:
                self.streams.popitem(last=False)
                self.stats["evicted_streams"] += 1
            stream = self._new_stream(proto, direction, client, server, conn_key)
            self.streams[key] = stream
        else:
            self.streams.move_to_end(key)

        messages: List[Message] = []
        if seg.flags & 0x02:  # SYN: a fresh connection
            stream.buf.clear()
            stream.next_seq = (seg.seq + 1) & SEQ_MASK

        data = seg.payload
        if data:
            if stream.next_seq is None:
                stream.next_seq = seg.seq
            diff = (seg.seq - stream.next_seq) & SEQ_MASK
            if diff == 0:
                pass
            elif diff >= 0x80000000:  # retransmission or overlap
                skip = (stream.next_seq - seg.seq) & SEQ_MASK
                data = b"" if skip >= len(data) else data[skip:]
                self.stats["retransmissions"] += 1
            else:  # gap: data was lost, so the stream cannot be trusted
                stream.buf.clear()
                stream.next_seq = seg.seq
                self.stats["gaps"] += 1
            if data:
                stream.buf += data
                stream.next_seq = (stream.next_seq + len(data)) & SEQ_MASK
                parsed, used = stream.parser.parse(stream.buf, seg.ts, client, server)  # type: ignore[attr-defined]
                if used:
                    del stream.buf[:used]
                messages.extend(parsed)
                if len(stream.buf) > stream.max_buffer:
                    stream.buf.clear()
                    self.stats["buffer_overflow"] += 1
                    messages.append(Message(
                        ts=seg.ts, proto=proto, client=client, server=server, direction=direction,
                        kind="invalid", size=0, valid=False, problems=("buffer_overflow",),
                    ))

        if seg.flags & 0x05:  # FIN or RST: connection is over
            self.streams.pop(key, None)
            self.connections.pop(conn_key, None)
        self.stats["messages"] += len(messages)
        return messages


def iter_messages(
    path,
    modbus_ports: Sequence[int] = (502,),
    mqtt_ports: Sequence[int] = (1883,),
    stats: Optional[Counter] = None,
) -> Iterator[Message]:
    """Yield messages from a capture file in capture order."""
    reassembler = StreamReassembler(modbus_ports, mqtt_ports)
    for segment in read_segments(path, stats):
        yield from reassembler.feed(segment)
    if stats is not None:
        stats.update(reassembler.stats)


def load_messages(
    path,
    modbus_ports: Sequence[int] = (502,),
    mqtt_ports: Sequence[int] = (1883,),
    stats: Optional[Counter] = None,
) -> List[Message]:
    """Read every message in a capture file into a list."""
    return list(iter_messages(path, modbus_ports, mqtt_ports, stats))


# --------------------------------------------------------------------------
# Writing captures (used by the simulator and the tests)
# --------------------------------------------------------------------------
def _checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    total = sum(struct.unpack("!%dH" % (len(data) // 2), data))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


def build_frame(seg: Segment) -> bytes:
    """Build an Ethernet/IPv4/TCP frame (IPv4 only) with valid checksums."""
    src_ip, dst_ip = socket.inet_aton(seg.src), socket.inet_aton(seg.dst)
    tcp_len = 20 + len(seg.payload)
    tcp = struct.pack("!HHIIBBHHH", seg.sport, seg.dport, seg.seq & SEQ_MASK, 1, 5 << 4, seg.flags, 8192, 0, 0)
    pseudo = src_ip + dst_ip + struct.pack("!BBH", 0, 6, tcp_len)
    tcp = tcp[:16] + struct.pack("!H", _checksum(pseudo + tcp + seg.payload)) + tcp[18:]
    ip = struct.pack("!BBHHHBBH", 0x45, 0, 20 + tcp_len, 0, 0x4000, 64, 6, 0) + src_ip + dst_ip
    ip = ip[:10] + struct.pack("!H", _checksum(ip)) + ip[12:]
    ether = b"\x02\x00\x00\x00\x00\x02" + b"\x02\x00\x00\x00\x00\x01" + b"\x08\x00"
    return ether + ip + tcp + seg.payload


def write_pcap(path, segments: Iterable[Segment]) -> int:
    """Write segments to a classic little-endian pcap file. Returns the packet count."""
    count = 0
    with open(path, "wb") as fh:
        fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for seg in segments:
            frame = build_frame(seg)
            sec = int(seg.ts)
            usec = int(round((seg.ts - sec) * 1_000_000))
            if usec >= 1_000_000:
                sec, usec = sec + 1, usec - 1_000_000
            fh.write(struct.pack("<IIII", sec, usec, len(frame), len(frame)))
            fh.write(frame)
            count += 1
    return count
