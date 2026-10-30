import gzip
import struct

import pytest

from protoguard.capture import (
    Segment,
    StreamReassembler,
    build_frame,
    decode_frame,
    iter_frames,
    load_messages,
    read_segments,
    write_pcap,
)
from protoguard.errors import CaptureError
from protoguard.messages import REQUEST, RESPONSE
from protoguard.parsers import modbus, mqtt

CLIENT, SERVER = "10.0.0.1", "10.0.0.2"


def req(payload, seq, ts=1.0, flags=0x18, sport=40000, dport=502):
    return Segment(ts, CLIENT, sport, SERVER, dport, seq, flags, payload)


def rsp(payload, seq, ts=1.0, flags=0x18, sport=502, dport=40000):
    return Segment(ts, SERVER, sport, CLIENT, dport, seq, flags, payload)


FRAME = modbus.read_request(1, 1, 3, 0, 10)  # 12 bytes


# ------------------------------------------------------------ reassembly
def test_message_split_across_segments_completes_on_the_last_one():
    r = StreamReassembler()
    assert r.feed(req(FRAME[:5], 1000, ts=1.0)) == []
    (m,) = r.feed(req(FRAME[5:], 1005, ts=2.0))
    assert m.ts == 2.0 and m.function == 3 and (m.client, m.server) == (CLIENT, SERVER)


def test_two_messages_in_one_segment():
    assert len(StreamReassembler().feed(req(FRAME + FRAME, 1000))) == 2


def test_response_direction_assigns_client_and_server():
    (m,) = StreamReassembler().feed(rsp(modbus.read_response(1, 1, 3, [1]), 5))
    assert m.direction == RESPONSE and (m.client, m.server) == (CLIENT, SERVER)


def test_retransmission_is_ignored_and_overlap_is_trimmed():
    r = StreamReassembler()
    assert len(r.feed(req(FRAME, 1000))) == 1
    assert r.feed(req(FRAME, 1000)) == []
    assert r.stats["retransmissions"] == 1

    r = StreamReassembler()
    assert r.feed(req(FRAME[:5], 1000)) == []
    (m,) = r.feed(req(FRAME, 1000))  # resend from the start, only the new tail is used
    assert m.function == 3


def test_gap_resets_the_stream():
    r = StreamReassembler()
    r.feed(req(FRAME, 1000))
    msgs = r.feed(req(FRAME, 5000))  # bytes in between were lost
    assert r.stats["gaps"] == 1 and len(msgs) == 1


def test_sequence_number_wraparound():
    r = StreamReassembler()
    assert r.feed(req(FRAME[:6], 0xFFFFFFFD)) == []
    (m,) = r.feed(req(FRAME[6:], (0xFFFFFFFD + 6) & 0xFFFFFFFF))
    assert m.function == 3


def test_fin_removes_stream_state():
    r = StreamReassembler()
    r.feed(req(FRAME, 1000, flags=0x11))
    assert not r.streams


def test_unrelated_ports_are_ignored_and_port_clash_rejected():
    r = StreamReassembler()
    assert r.feed(req(FRAME, 1, dport=8080)) == []
    assert r.stats["ignored_segments"] == 1
    with pytest.raises(ValueError):
        StreamReassembler(modbus_ports=(1883,), mqtt_ports=(1883,))


def test_custom_ports():
    r = StreamReassembler(modbus_ports=(1502,))
    assert len(r.feed(req(FRAME, 1, dport=1502))) == 1


def test_mqtt_protocol_level_is_shared_between_directions():
    r = StreamReassembler()
    r.feed(Segment(1.0, CLIENT, 40000, SERVER, 1883, 100, 0x18, mqtt.connect("c", level=5)))
    body = mqtt.encode_string("t") + b"\x00" + b"abc"
    (m,) = r.feed(Segment(2.0, SERVER, 1883, CLIENT, 40000, 900, 0x18, mqtt.packet(3, 0, body)))
    assert m.payload_len == 3 and m.direction == RESPONSE


def test_stream_table_is_bounded():
    r = StreamReassembler(max_streams=3)
    for port in range(40000, 40010):
        r.feed(req(b"", 1, sport=port))
    assert len(r.streams) == 3 and r.stats["evicted_streams"] == 7


def test_buffer_overflow_is_reported_not_unbounded():
    r = StreamReassembler()
    # a "message" that claims to be 254 bytes but never completes, followed by endless junk
    stuck = struct.pack(">HHHB", 1, 0, 254, 1)
    out, seq = [], 0
    out += r.feed(req(stuck, seq))
    seq += len(stuck)
    for _ in range(2000):
        out += r.feed(req(b"\x00" * 200, seq))
        seq += 200
    assert r.stats["buffer_overflow"] >= 1 or any(not m.valid for m in out) or len(r.streams[next(iter(r.streams))].buf) < 4096 + 400


# ------------------------------------------------------------ frame codec
def test_frame_round_trip_and_ethernet_padding():
    seg = Segment(5.5, CLIENT, 40000, SERVER, 502, 123456, 0x18, b"abc")
    frame = build_frame(seg)
    back = decode_frame(5.5, 1, frame)
    assert (back.src, back.sport, back.dst, back.dport, back.seq, back.payload) == (CLIENT, 40000, SERVER, 502, 123456, b"abc")
    assert decode_frame(5.5, 1, frame + b"\x00" * 6).payload == b"abc"  # trailer padding is not payload


def test_vlan_tag_is_skipped():
    frame = build_frame(Segment(1.0, CLIENT, 1, SERVER, 502, 1, 0x18, b"x"))
    tagged = frame[:12] + b"\x81\x00\x00\x05" + frame[12:]
    assert decode_frame(1.0, 1, tagged).payload == b"x"


def test_non_tcp_fragments_and_junk_are_skipped():
    frame = bytearray(build_frame(Segment(1.0, CLIENT, 1, SERVER, 502, 1, 0x18, b"x")))
    udp = bytearray(frame)
    udp[14 + 9] = 17
    assert decode_frame(1.0, 1, bytes(udp)) is None
    fragment = bytearray(frame)
    fragment[14 + 6 : 14 + 8] = b"\x20\x00"  # more-fragments flag
    assert decode_frame(1.0, 1, bytes(fragment)) is None
    assert decode_frame(1.0, 1, b"\x00" * 5) is None
    assert decode_frame(1.0, 999, bytes(frame)) is None


def test_raw_ip_link_type():
    frame = build_frame(Segment(1.0, CLIENT, 1, SERVER, 502, 1, 0x18, b"x"))
    assert decode_frame(1.0, 101, frame[14:]).payload == b"x"


# ------------------------------------------------------------ file formats
def _segments():
    return [req(FRAME, 1000, ts=1_700_000_000.25), rsp(modbus.read_response(1, 1, 3, [7] * 10), 5, ts=1_700_000_000.5)]


def test_pcap_write_and_read_round_trip(tmp_path):
    path = tmp_path / "x.pcap"
    assert write_pcap(path, _segments()) == 2
    back = list(read_segments(path))
    assert [s.payload for s in back] == [s.payload for s in _segments()]
    assert back[0].ts == pytest.approx(1_700_000_000.25, abs=1e-6)
    msgs = load_messages(path)
    assert [m.direction for m in msgs] == [REQUEST, RESPONSE]


def test_gzip_capture(tmp_path):
    plain = tmp_path / "x.pcap"
    write_pcap(plain, _segments())
    packed = tmp_path / "x.pcap.gz"
    packed.write_bytes(gzip.compress(plain.read_bytes()))
    assert len(load_messages(packed)) == 2


def test_big_endian_pcap(tmp_path):
    frame = build_frame(_segments()[0])
    data = struct.pack(">IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    data += struct.pack(">IIII", 1_700_000_000, 250_000, len(frame), len(frame)) + frame
    path = tmp_path / "be.pcap"
    path.write_bytes(data)
    assert len(load_messages(path)) == 1


def _pcapng_block(btype, body):
    pad = (-len(body)) % 4
    total = 12 + len(body) + pad
    return struct.pack("<II", btype, total) + body + b"\x00" * pad + struct.pack("<I", total)


def test_pcapng_with_nanosecond_resolution(tmp_path):
    frame = build_frame(_segments()[0])
    shb = _pcapng_block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    options = struct.pack("<HHB", 9, 1, 9) + b"\x00\x00\x00" + struct.pack("<HH", 0, 0)  # if_tsresol = 10^-9
    idb = _pcapng_block(1, struct.pack("<HHI", 1, 0, 65535) + options)
    t_ns = 1_700_000_000_250_000_000
    epb = _pcapng_block(6, struct.pack("<IIIII", 0, t_ns >> 32, t_ns & 0xFFFFFFFF, len(frame), len(frame)) + frame)
    path = tmp_path / "x.pcapng"
    frame2 = build_frame(_segments()[1])  # a different segment (the same one twice would be a retransmission)
    epb2 = _pcapng_block(6, struct.pack("<IIIII", 0, t_ns >> 32, t_ns & 0xFFFFFFFF, len(frame2), len(frame2)) + frame2)
    path.write_bytes(shb + idb + epb + epb2)
    frames = list(iter_frames(path))
    assert len(frames) == 2 and frames[0][0] == pytest.approx(1_700_000_000.25, abs=1e-6)
    assert len(load_messages(path)) == 2


def test_bad_inputs_raise_capture_error(tmp_path):
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"this is not a capture file")
    with pytest.raises(CaptureError):
        list(iter_frames(junk))
    with pytest.raises(CaptureError):
        list(iter_frames(tmp_path / "missing.pcap"))
    short = tmp_path / "short.pcap"
    short.write_bytes(b"\xd4\xc3\xb2\xa1" + b"\x00" * 6)
    with pytest.raises(CaptureError):
        list(iter_frames(short))


def test_truncated_final_record_is_tolerated(tmp_path):
    path = tmp_path / "x.pcap"
    write_pcap(path, _segments())
    data = path.read_bytes()
    path.write_bytes(data[:-10])
    assert len(load_messages(path)) == 1
