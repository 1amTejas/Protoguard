import struct

import pytest

from protoguard.messages import REQUEST, RESPONSE
from protoguard.parsers import ModbusParser, MqttParser, modbus, mqtt

C, S = "10.0.0.1", "10.0.0.2"


def mb(data, direction=REQUEST):
    buf = bytearray(data)
    msgs, used = ModbusParser(direction).parse(buf, 1.0, C, S)
    return msgs, used


def mq(data, direction=REQUEST, shared=None, **kwargs):
    buf = bytearray(data)
    msgs, used = MqttParser(direction, shared, **kwargs).parse(buf, 1.0, C, S)
    return msgs, used


# ------------------------------------------------------------------ Modbus
def test_modbus_read_request_and_response():
    (m,), used = mb(modbus.read_request(7, 1, 3, 100, 10))
    assert (m.kind, m.function, m.address, m.quantity, m.unit_id, m.size, m.valid) == ("fc3", 3, 100, 10, 1, 12, True)
    assert used == 12
    (r,), _ = mb(modbus.read_response(7, 1, 3, [1, 2, 3]), RESPONSE)
    assert r.valid and r.function == 3 and r.direction == RESPONSE


def test_modbus_writes():
    (m,), _ = mb(modbus.write_single_request(1, 1, 6, 200, 99))
    assert (m.function, m.address, m.quantity, m.valid) == (6, 200, 1, True)
    (m,), _ = mb(modbus.write_multiple_request(1, 1, 50, [1, 2, 3]))
    assert (m.function, m.address, m.quantity, m.valid) == (16, 50, 3, True)
    (r,), _ = mb(modbus.write_multiple_response(1, 1, 50, 3), RESPONSE)
    assert (r.address, r.quantity, r.valid) == (50, 3, True)
    (bad,), _ = mb(modbus.write_single_request(1, 1, 5, 10, 0x1234))
    assert "bad_coil_value" in bad.problems and not bad.valid


def test_modbus_exceptions():
    (r,), _ = mb(modbus.exception_response(1, 1, 3, 2), RESPONSE)
    assert r.exception == 2 and r.kind == "fc3_exc" and r.function == 3 and r.valid
    (q,), _ = mb(modbus.exception_response(1, 1, 3, 2), REQUEST)
    assert "exception_in_request" in q.problems


@pytest.mark.parametrize(
    "frame, problem",
    [
        (modbus.raw_request(1, 1, 100), "unknown_function"),
        (modbus.read_request(1, 1, 3, 0, 200), "bad_quantity"),
        (modbus.read_request(1, 1, 3, 65535, 10), "address_overflow"),
        (modbus.raw_request(1, 1, 3, b"\x00"), "bad_length"),
    ],
)
def test_modbus_spec_violations(frame, problem):
    (m,), _ = mb(frame)
    assert problem in m.problems and not m.valid


@pytest.mark.parametrize(
    "header",
    [
        struct.pack(">HHHB", 1, 1, 6, 1),  # protocol id must be 0
        struct.pack(">HHHB", 1, 0, 1, 1),  # length too small
        struct.pack(">HHHB", 1, 0, 400, 1),  # length too large
    ],
)
def test_modbus_bad_mbap_drops_buffer(header):
    msgs, used = mb(header + b"\x03\x00\x00\x00\x0a")
    assert len(msgs) == 1 and msgs[0].problems == ("bad_mbap",) and used == len(header) + 5


def test_modbus_partial_and_multiple_frames():
    frame = modbus.read_request(1, 1, 3, 0, 10)
    assert mb(frame[:5]) == ([], 0)
    assert mb(frame[:11]) == ([], 0)
    msgs, used = mb(frame + frame + frame[:4])
    assert len(msgs) == 2 and used == 2 * len(frame)


# -------------------------------------------------------------------- MQTT
def test_mqtt_connect_and_basic_packets():
    (m,), _ = mq(mqtt.connect("dev-1"))
    assert (m.kind, m.client_id, m.valid) == ("CONNECT", "dev-1", True)
    for data, direction, kind in [
        (mqtt.connack(), RESPONSE, "CONNACK"),
        (mqtt.puback(5), RESPONSE, "PUBACK"),
        (mqtt.pingreq(), REQUEST, "PINGREQ"),
        (mqtt.pingresp(), RESPONSE, "PINGRESP"),
        (mqtt.disconnect(), REQUEST, "DISCONNECT"),
        (mqtt.suback(1, [0]), RESPONSE, "SUBACK"),
    ]:
        (m,), _ = mq(data, direction)
        assert m.kind == kind and m.valid, kind


def test_mqtt_publish_fields():
    (m,), _ = mq(mqtt.publish("a/b", b"hello world", qos=1, retain=True, packet_id=9))
    assert (m.topic, m.qos, m.retain, m.payload_len, m.valid) == ("a/b", 1, True, 11, True)
    assert m.payload_entropy > 0
    (empty,), _ = mq(mqtt.publish("a/b", b""))
    assert empty.payload_len == 0 and empty.payload_entropy == 0.0


def test_mqtt_subscribe_and_unsubscribe():
    (m,), _ = mq(mqtt.subscribe(1, [("a/#", 0), ("b/+", 1)]))
    assert m.filters == ("a/#", "b/+") and m.valid
    (u,), _ = mq(mqtt.packet(10, 2, struct.pack(">H", 1) + mqtt.encode_string("a/#")))
    assert u.kind == "UNSUBSCRIBE" and u.filters == ("a/#",) and u.valid


def test_mqtt_v5_properties_are_skipped_once_connect_is_seen():
    shared = {"version": 4}
    mq(mqtt.connect("c", level=5), shared=shared)
    assert shared["version"] == 5
    body = mqtt.encode_string("t") + b"\x00" + b"abc"  # empty properties, 3-byte payload
    (m,), _ = mq(mqtt.packet(3, 0, body), RESPONSE, shared=shared)
    assert m.payload_len == 3 and m.valid
    # the same bytes read as 3.1.1 would count the property byte as payload
    (m4,), _ = mq(mqtt.packet(3, 0, body))
    assert m4.payload_len == 4


@pytest.mark.parametrize(
    "data, direction, problem",
    [
        (mqtt.packet(3, 0b0110, mqtt.encode_string("a") + b"x"), REQUEST, "bad_qos"),
        (mqtt.packet(3, 0, mqtt.encode_string("a/#") + b"x"), REQUEST, "wildcard_in_publish"),
        (mqtt.packet(3, 0, mqtt.encode_string("") + b"x"), REQUEST, "empty_topic"),
        (mqtt.packet(8, 0, struct.pack(">H", 1) + mqtt.encode_string("x") + b"\x00"), REQUEST, "bad_flags"),
        (mqtt.connect("c"), RESPONSE, "wrong_direction"),
        (mqtt.connack(), REQUEST, "wrong_direction"),
        (mqtt.packet(3, 0, b"\x00\x10ab"), REQUEST, "truncated_body"),
        (mqtt.packet(8, 2, struct.pack(">H", 1)), REQUEST, "empty_subscribe"),
    ],
)
def test_mqtt_spec_violations(data, direction, problem):
    (m,), _ = mq(data, direction)
    assert problem in m.problems and not m.valid


@pytest.mark.parametrize(
    "data, problem",
    [
        (b"\x00\x00", "reserved_type"),
        (b"\x30\xff\xff\xff\xff\x01", "bad_remaining_length"),
    ],
)
def test_mqtt_malformed_drops_buffer(data, problem):
    msgs, used = mq(data)
    assert msgs[0].problems == (problem,) and used == len(data)


def test_mqtt_oversize_and_partial_and_multiple():
    big = mqtt.publish("a", b"x" * 200)
    msgs, used = mq(big, max_packet=100)
    assert msgs[0].problems == ("oversize",) and used == len(big)
    assert mq(big[:50]) == ([], 0)
    msgs, used = mq(big + big + big[:3])
    assert len(msgs) == 2 and used == 2 * len(big)


def test_mqtt_varint_encoding_boundaries():
    assert [len(mqtt.encode_varint(n)) for n in (0, 127, 128, 16383, 16384, 2097152, 268435455)] == [1, 1, 2, 2, 3, 4, 4]
    with pytest.raises(ValueError):
        mqtt.encode_varint(268435456)
    (m,), _ = mq(mqtt.publish("a", b"x" * 20000))
    assert m.payload_len == 20000
