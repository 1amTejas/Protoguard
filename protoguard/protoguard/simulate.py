"""Synthetic labelled traffic for demos, tests and pipeline checks.

IMPORTANT: this generates packets into pcap files only. It never opens a
socket and never sends anything on a network.

The simulated plant has Modbus/TCP polling (a master and an HMI reading two
PLCs, plus occasional setpoint writes) and an MQTT broker with sensors
publishing telemetry and a dashboard subscribed to all of it. The attack
catalogue covers floods, scans, malformed frames, oversized payloads and a few
deliberately quiet "insider" cases that reuse a legitimate client's address.

Results on this data show that the pipeline works end to end. They say
little about real networks, whose traffic is messier and whose attacks are
not drawn from a list the detector's author wrote. Evaluate on real captures
before drawing any conclusion.
"""
from __future__ import annotations

import base64
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .capture import Segment, write_pcap
from .labels import Interval, save_labels
from .parsers import modbus, mqtt

MASTER, HMI = "192.168.10.10", "192.168.10.11"
PLC1, PLC2 = "192.168.10.20", "192.168.10.21"
MODBUS_ATTACKER = "192.168.10.99"
BROKER = "192.168.20.5"
SENSORS = ["192.168.20.11", "192.168.20.12", "192.168.20.13", "192.168.20.14", "192.168.20.15"]
DASHBOARD = "192.168.20.30"
MQTT_ATTACKER = "192.168.20.99"

ATTACKS = [
    "modbus_write_flood", "modbus_function_scan", "modbus_read_flood", "modbus_malformed",
    "modbus_setpoint_tamper", "mqtt_publish_flood", "mqtt_connect_flood", "mqtt_topic_scan",
    "mqtt_oversized_payload", "mqtt_malformed", "mqtt_payload_injection",
]


@dataclass
class Event:
    ts: float
    client: str
    cport: int
    server: str
    sport: int
    c2s: bool  # True: client -> server
    payload: bytes


class Simulator:
    def __init__(self, seed: int, start: float, duration: float) -> None:
        self.rng = random.Random(seed)
        self.start = start
        self.end = start + duration
        self.events: List[Event] = []
        self.intervals: List[Interval] = []
        self._port = 20000 + self.rng.randrange(10000)
        self._tids: Dict[Tuple[str, int], int] = {}

    # --------------------------------------------------------------- basics
    def new_port(self) -> int:
        self._port = self._port + 1 if self._port < 65000 else 20000
        return self._port

    def send(self, ts: float, client: str, cport: int, server: str, sport: int, c2s: bool, payload: bytes) -> None:
        self.events.append(Event(ts, client, cport, server, sport, c2s, payload))

    def _tid(self, key: Tuple[str, int]) -> int:
        self._tids[key] = (self._tids.get(key, self.rng.randrange(65536)) + 1) & 0xFFFF
        return self._tids[key]

    def _interval(self, attack: str, start: float, duration: float, ip: str, target: str) -> None:
        # 0.5 s of slack: replies to the last request arrive just after the attack "ends".
        self.intervals.append(Interval(start, start + duration + 0.5, attack, ip, target))

    # --------------------------------------------------------- normal Modbus
    def _mb_exchange(self, ts: float, client: str, cport: int, plc: str, request: bytes, response: Optional[bytes]) -> None:
        self.send(ts, client, cport, plc, 502, True, request)
        if response is not None:
            self.send(ts + self.rng.uniform(0.004, 0.03), client, cport, plc, 502, False, response)

    def _mb_poll(self, client: str, cport: int, plc: str, unit: int, fc: int, address: int, qty: int, period: float) -> None:
        t = self.start + self.rng.uniform(0, period)
        while t < self.end:
            tid = self._tid((client, cport))
            request = modbus.read_request(tid, unit, fc, address, qty)
            if self.rng.random() < 0.003:  # a device occasionally answers "busy"
                response = modbus.exception_response(tid, unit, fc, 6)
            else:
                response = modbus.read_response(tid, unit, fc, [self.rng.randrange(0, 1000) for _ in range(qty)])
            self._mb_exchange(t, client, cport, plc, request, response)
            t += period * self.rng.uniform(0.9, 1.1)

    def _mb_setpoint_writes(self, client: str, cport: int, plc: str, unit: int, mean_interval: float) -> None:
        t = self.start + self.rng.expovariate(1 / mean_interval)
        while t < self.end:
            tid = self._tid((client, cport))
            address, value = self.rng.choice((200, 201, 202)), self.rng.randrange(50, 150)
            self._mb_exchange(
                t, client, cport, plc,
                modbus.write_single_request(tid, unit, 6, address, value),
                modbus.write_single_response(tid, unit, 6, address, value),
            )
            t += self.rng.expovariate(1 / mean_interval)

    def normal_modbus(self) -> None:
        c1, c2, c3 = self.new_port(), self.new_port(), self.new_port()
        self._mb_poll(MASTER, c1, PLC1, 1, 3, 0, 10, 0.5)
        self._mb_poll(MASTER, c1, PLC1, 1, 4, 100, 4, 2.0)
        self._mb_setpoint_writes(MASTER, c1, PLC1, 1, 20.0)
        self._mb_poll(MASTER, c2, PLC2, 2, 3, 0, 8, 1.0)
        self._mb_poll(HMI, c3, PLC1, 1, 3, 0, 10, 2.0)

    # ----------------------------------------------------------- normal MQTT
    def _mq_connect(self, ts: float, client: str, client_id: str) -> int:
        port = self.new_port()
        self.send(ts, client, port, BROKER, 1883, True, mqtt.connect(client_id))
        self.send(ts + self.rng.uniform(0.002, 0.02), client, port, BROKER, 1883, False, mqtt.connack())
        return port

    def _mq_pings(self, client: str, port: int, t0: float, interval: float) -> None:
        t = t0 + interval * self.rng.uniform(0.9, 1.1)
        while t < self.end:
            self.send(t, client, port, BROKER, 1883, True, mqtt.pingreq())
            self.send(t + 0.01, client, port, BROKER, 1883, False, mqtt.pingresp())
            t += interval * self.rng.uniform(0.95, 1.05)

    def _mq_sensor(self, ip: str, index: int, dashboard_port: int) -> None:
        client_id = f"sensor-{index + 1}"
        topic = f"factory/line1/sensor{index + 1}/temp"
        qos = 1 if index >= 3 else 0
        t = self.start + self.rng.uniform(0.3, 1.5)
        port = self._mq_connect(t, ip, client_id)
        next_reconnect = t + self.rng.expovariate(1 / 600.0)
        next_ping = t + 30.0
        value, packet_id = 20.0 + self.rng.uniform(-2, 2), 0
        while True:
            t += 2.0 * self.rng.uniform(0.9, 1.1)
            if t >= self.end:
                break
            if t >= next_reconnect:  # sensors reconnect now and then
                self.send(t, ip, port, BROKER, 1883, True, mqtt.disconnect())
                port = self._mq_connect(t + 0.05, ip, client_id)
                next_reconnect = t + self.rng.expovariate(1 / 600.0)
                next_ping = t + 30.0
                continue
            value += self.rng.uniform(-0.3, 0.3)
            payload = ('{"t":%.2f,"u":"C"}' % value).encode()
            packet_id = packet_id % 65535 + 1
            self.send(t, ip, port, BROKER, 1883, True, mqtt.publish(topic, payload, qos, False, packet_id))
            if qos:
                self.send(t + 0.005, ip, port, BROKER, 1883, False, mqtt.puback(packet_id))
            self.send(t + 0.01, DASHBOARD, dashboard_port, BROKER, 1883, False, mqtt.publish(topic, payload, 0))
            if t >= next_ping:
                self.send(t + 0.02, ip, port, BROKER, 1883, True, mqtt.pingreq())
                self.send(t + 0.03, ip, port, BROKER, 1883, False, mqtt.pingresp())
                next_ping = t + 30.0

    def normal_mqtt(self) -> None:
        port = self._mq_connect(self.start + 0.2, DASHBOARD, "dashboard-1")
        t = self.start + 0.3
        self.send(t, DASHBOARD, port, BROKER, 1883, True, mqtt.subscribe(1, [("factory/line1/#", 0)]))
        self.send(t + 0.01, DASHBOARD, port, BROKER, 1883, False, mqtt.suback(1, [0]))
        self._mq_pings(DASHBOARD, port, t, 30.0)
        for index, ip in enumerate(SENSORS):
            self._mq_sensor(ip, index, port)

    def normal(self) -> None:
        self.normal_modbus()
        self.normal_mqtt()

    # ------------------------------------------------------- Modbus attacks
    def attack_modbus_write_flood(self, start: float, duration: float = 20.0, ip: str = MODBUS_ATTACKER) -> None:
        port, t = self.new_port(), start
        while t < start + duration:
            tid = self._tid((ip, port))
            address = self.rng.randrange(0, 400)
            if self.rng.random() < 0.7:
                value = self.rng.randrange(0, 65536)
                request = modbus.write_single_request(tid, 1, 6, address, value)
                response = modbus.write_single_response(tid, 1, 6, address, value)
            else:
                values = [self.rng.randrange(65536) for _ in range(4)]
                request = modbus.write_multiple_request(tid, 1, address, values)
                response = modbus.write_multiple_response(tid, 1, address, 4)
            self._mb_exchange(t, ip, port, PLC1, request, response)
            t += self.rng.expovariate(40.0)
        self._interval("modbus_write_flood", start, duration, ip, PLC1)

    def attack_modbus_function_scan(self, start: float, duration: float = 25.0, ip: str = MODBUS_ATTACKER) -> None:
        port, t = self.new_port(), start
        while t < start + duration:
            tid = self._tid((ip, port))
            function, unit = self.rng.randrange(1, 128), self.rng.randrange(0, 248)
            request = modbus.raw_request(tid, unit, function, self.rng.randbytes(self.rng.randrange(0, 5)))
            response = modbus.exception_response(tid, unit, function, 1) if self.rng.random() < 0.8 else None
            self._mb_exchange(t, ip, port, PLC1, request, response)
            t += self.rng.expovariate(6.0)
        self._interval("modbus_function_scan", start, duration, ip, PLC1)

    def attack_modbus_read_flood(self, start: float, duration: float = 15.0, ip: str = MASTER) -> None:
        port, t = self.new_port(), start
        while t < start + duration:
            tid = self._tid((ip, port))
            request = modbus.read_request(tid, 1, 3, 0, 125)
            response = None
            if self.rng.random() < 0.6:  # the PLC cannot keep up
                response = modbus.read_response(tid, 1, 3, [self.rng.randrange(0, 1000) for _ in range(125)])
            self._mb_exchange(t, ip, port, PLC1, request, response)
            t += self.rng.expovariate(150.0)
        self._interval("modbus_read_flood", start, duration, ip, PLC1)

    def attack_modbus_malformed(self, start: float, duration: float = 15.0, ip: str = MODBUS_ATTACKER) -> None:
        port, t = self.new_port(), start
        while t < start + duration:
            tid = self._tid((ip, port))
            kind = self.rng.randrange(3)
            if kind == 0:
                frame = struct.pack(">HHHB", tid, 0x1234, 6, 1) + bytes([3, 0, 0, 0, 10])  # bad protocol id
            elif kind == 1:
                frame = struct.pack(">HHHB", tid, 0, 0, 1) + b"\x03\x00"  # impossible length
            else:
                frame = struct.pack(">HHHB", tid, 0, 300, 1) + self.rng.randbytes(8)  # length too large
            self.send(t, ip, port, PLC1, 502, True, frame)
            t += self.rng.expovariate(3.0)
        self._interval("modbus_malformed", start, duration, ip, PLC1)

    def attack_modbus_setpoint_tamper(self, start: float, duration: float = 40.0, ip: str = MASTER) -> None:
        """Quiet insider case: valid writes to a known address with bad values, a few per minute."""
        port, t = self.new_port(), start
        while t < start + duration:
            tid = self._tid((ip, port))
            value = self.rng.choice((0, 65535, 9999))
            self._mb_exchange(
                t, ip, port, PLC1,
                modbus.write_single_request(tid, 1, 6, 200, value),
                modbus.write_single_response(tid, 1, 6, 200, value),
            )
            t += self.rng.uniform(2.0, 4.0)
        self._interval("modbus_setpoint_tamper", start, duration, ip, PLC1)

    # --------------------------------------------------------- MQTT attacks
    def attack_mqtt_publish_flood(self, start: float, duration: float = 20.0, ip: str = MQTT_ATTACKER) -> None:
        port = self._mq_connect(start, ip, "attacker-flood")
        t = start + 0.1
        while t < start + duration:
            topic = f"factory/line1/sensor{self.rng.randrange(1, 6)}/temp"
            payload = ('{"t":%.2f,"u":"C"}' % self.rng.uniform(-50, 150)).encode()
            self.send(t, ip, port, BROKER, 1883, True, mqtt.publish(topic, payload, 0))
            t += self.rng.expovariate(100.0)
        self._interval("mqtt_publish_flood", start, duration, ip, BROKER)

    def attack_mqtt_connect_flood(self, start: float, duration: float = 20.0, ip: str = MQTT_ATTACKER) -> None:
        t = start
        while t < start + duration:
            self._mq_connect(t, ip, "c%08x" % self.rng.getrandbits(32))
            t += self.rng.expovariate(25.0)
        self._interval("mqtt_connect_flood", start, duration, ip, BROKER)

    def attack_mqtt_topic_scan(self, start: float, duration: float = 20.0, ip: str = MQTT_ATTACKER) -> None:
        port = self._mq_connect(start, ip, "attacker-scan")
        words = ["admin", "config", "firmware", "secret", "status", "cmd", "debug", "keys"]
        t, packet_id = start + 0.2, 0
        filters = ["#", "$SYS/#"]
        while t < start + duration:
            if filters:
                topic = filters.pop(0)
            else:
                topic = f"factory/line{self.rng.randrange(1, 4)}/{self.rng.choice(words)}"
            packet_id = packet_id % 65535 + 1
            self.send(t, ip, port, BROKER, 1883, True, mqtt.subscribe(packet_id, [(topic, 0)]))
            self.send(t + 0.01, ip, port, BROKER, 1883, False, mqtt.suback(packet_id, [0x80]))
            t += self.rng.expovariate(5.0)
        self._interval("mqtt_topic_scan", start, duration, ip, BROKER)

    def attack_mqtt_oversized_payload(self, start: float, duration: float = 15.0, ip: str = SENSORS[1]) -> None:
        """Insider case: a real sensor's address sending huge random payloads."""
        port = self._mq_connect(start, ip, "sensor-2")
        t = start + 0.1
        while t < start + duration:
            payload = self.rng.randbytes(48 * 1024)
            self.send(t, ip, port, BROKER, 1883, True, mqtt.publish("factory/line1/sensor2/temp", payload, 0))
            t += self.rng.expovariate(2.0)
        self._interval("mqtt_oversized_payload", start, duration, ip, BROKER)

    def attack_mqtt_malformed(self, start: float, duration: float = 15.0, ip: str = MQTT_ATTACKER) -> None:
        port, t = self.new_port(), start
        while t < start + duration:
            kind = self.rng.randrange(3)
            if kind == 0:
                packet = b"\x00\x00"  # reserved packet type
            elif kind == 1:
                packet = mqtt.packet(3, 0b0110, mqtt.encode_string("a/b") + b"x")  # QoS 3
            else:
                packet = mqtt.packet(8, 0, struct.pack(">H", 1) + mqtt.encode_string("x/y") + b"\x00")  # bad flags
            self.send(t, ip, port, BROKER, 1883, True, packet)
            t += self.rng.expovariate(3.0)
        self._interval("mqtt_malformed", start, duration, ip, BROKER)

    def attack_mqtt_payload_injection(self, start: float, duration: float = 30.0, ip: str = SENSORS[2]) -> None:
        """Insider case: a real sensor's address publishing long, high-entropy payloads to its own topic."""
        port = self._mq_connect(start, ip, "sensor-3")
        t = start + 0.1
        while t < start + duration:
            payload = base64.b64encode(self.rng.randbytes(220))
            self.send(t, ip, port, BROKER, 1883, True, mqtt.publish("factory/line1/sensor3/temp", payload, 0))
            t += self.rng.uniform(0.3, 0.7)
        self._interval("mqtt_payload_injection", start, duration, ip, BROKER)

    def schedule_attacks(self, first: float = 150.0, spacing: float = 90.0) -> None:
        """Run every attack once, in a random order, one per time slot."""
        order = list(ATTACKS)
        self.rng.shuffle(order)
        t = self.start + first
        for name in order:
            if t + 45 > self.end:
                break
            getattr(self, f"attack_{name}")(t)
            t += spacing


def events_to_segments(events: List[Event], seed: int = 0, mss: int = 1400) -> List[Segment]:
    """Turn events into TCP segments with consistent sequence numbers, split at ``mss``."""
    rng = random.Random(seed)
    next_seq: Dict[tuple, int] = {}
    last_ts: Dict[tuple, float] = {}
    segments: List[Segment] = []
    for ev in sorted(events, key=lambda e: e.ts):
        if ev.c2s:
            src, sport, dst, dport = ev.client, ev.cport, ev.server, ev.sport
        else:
            src, sport, dst, dport = ev.server, ev.sport, ev.client, ev.cport
        key = (src, sport, dst, dport)
        seq = next_seq.get(key)
        if seq is None:
            seq = rng.getrandbits(32)
        ts = ev.ts
        for offset in range(0, max(1, len(ev.payload)), mss):
            chunk = ev.payload[offset : offset + mss]
            ts = max(ts, last_ts.get(key, 0.0) + 1e-6)  # keep each flow strictly ordered
            segments.append(Segment(ts, src, sport, dst, dport, seq, 0x18, chunk))
            last_ts[key] = ts
            seq = (seq + len(chunk)) & 0xFFFFFFFF
            ts += 1e-5
        next_seq[key] = seq
    segments.sort(key=lambda s: s.ts)
    return segments


def build_dataset(
    out_dir,
    seed: int = 1,
    train_duration: float = 1800.0,
    test_duration: float = 1200.0,
    start: float = 1_700_000_000.0,
) -> Dict[str, object]:
    """Write ``train_normal.pcap``, ``test.pcap`` and ``labels.csv`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = Simulator(seed, start, train_duration)
    train.normal()
    train_packets = write_pcap(out / "train_normal.pcap", events_to_segments(train.events, seed))

    test = Simulator(seed + 1, start + train_duration + 600.0, test_duration)
    test.normal()
    test.schedule_attacks()
    test_packets = write_pcap(out / "test.pcap", events_to_segments(test.events, seed + 1))
    save_labels(out / "labels.csv", test.intervals)

    return {
        "train_pcap": str(out / "train_normal.pcap"), "train_packets": train_packets,
        "test_pcap": str(out / "test.pcap"), "test_packets": test_packets,
        "labels": str(out / "labels.csv"), "attacks": len(test.intervals),
    }
