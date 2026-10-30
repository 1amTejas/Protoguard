"""Modbus/TCP parser and frame builders.

An ADU is the 7-byte MBAP header (transaction id, protocol id = 0, length, unit
id) followed by a PDU (function code + data). ``length`` counts the unit id plus
the PDU. The parser is stateless apart from the buffer it is handed, so it works
on reassembled TCP streams that contain several messages or a partial one.
"""
from __future__ import annotations

import struct
from typing import List, Optional, Tuple

from ..messages import REQUEST, Message

KNOWN_FUNCTIONS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 17, 20, 21, 22, 23, 24, 43})
READ_FUNCTIONS = frozenset({1, 2, 3, 4})
WRITE_FUNCTIONS = frozenset({5, 6, 15, 16, 22, 23})
MAX_LENGTH = 254  # unit id (1) + PDU (<= 253)


def category(function: Optional[int]) -> str:
    """"read", "write" or "other" for a function code."""
    if function in READ_FUNCTIONS:
        return "read"
    if function in WRITE_FUNCTIONS:
        return "write"
    return "other"


class ModbusParser:
    """Parse Modbus/TCP ADUs from one direction of a TCP stream."""

    proto = "modbus"

    def __init__(self, direction: str) -> None:
        self.direction = direction

    # ------------------------------------------------------------------ API
    def parse(self, buf: bytearray, ts: float, client: str, server: str) -> Tuple[List[Message], int]:
        """Return ``(messages, bytes_consumed)``. Incomplete trailing data is left unconsumed."""
        messages: List[Message] = []
        pos, n = 0, len(buf)
        while n - pos >= 7:
            _tid, pid, length, unit = struct.unpack_from(">HHHB", buf, pos)
            if pid != 0 or length < 2 or length > MAX_LENGTH:
                messages.append(self._bad(ts, client, server, n - pos, "bad_mbap"))
                return messages, n  # cannot resynchronise: drop the buffer
            total = 6 + length
            if n - pos < total:
                break
            pdu = bytes(buf[pos + 7 : pos + total])
            messages.append(self._decode(pdu, unit, total, ts, client, server))
            pos += total
        return messages, pos

    # ------------------------------------------------------------- internals
    def _bad(self, ts: float, client: str, server: str, size: int, problem: str) -> Message:
        return Message(
            ts=ts, proto="modbus", client=client, server=server, direction=self.direction,
            kind="invalid", size=size, valid=False, problems=(problem,),
        )

    def _decode(self, pdu: bytes, unit: int, size: int, ts: float, client: str, server: str) -> Message:
        raw_fc = pdu[0]
        is_exception = bool(raw_fc & 0x80)
        function = raw_fc & 0x7F
        problems: List[str] = []
        address: Optional[int] = None
        quantity: Optional[int] = None
        exception: Optional[int] = None

        if function not in KNOWN_FUNCTIONS:
            problems.append("unknown_function")
        if is_exception:
            if self.direction == REQUEST:
                problems.append("exception_in_request")
            if len(pdu) != 2:
                problems.append("bad_length")
            else:
                exception = pdu[1]
        elif self.direction == REQUEST:
            address, quantity = self._request_fields(function, pdu, problems)
        else:
            address, quantity = self._response_fields(function, pdu, problems)

        kind = f"fc{function}" + ("_exc" if is_exception else "")
        return Message(
            ts=ts, proto="modbus", client=client, server=server, direction=self.direction,
            kind=kind, size=size, valid=not problems, problems=tuple(problems),
            unit_id=unit, function=function, address=address, quantity=quantity, exception=exception,
        )

    @staticmethod
    def _request_fields(function: int, pdu: bytes, problems: List[str]) -> Tuple[Optional[int], Optional[int]]:
        n = len(pdu)
        if function in (1, 2, 3, 4):
            if n != 5:
                problems.append("bad_length")
                return None, None
            address, quantity = struct.unpack(">HH", pdu[1:5])
            limit = 2000 if function in (1, 2) else 125
            if not 1 <= quantity <= limit:
                problems.append("bad_quantity")
            if address + quantity > 65536:
                problems.append("address_overflow")
            return address, quantity
        if function in (5, 6):
            if n != 5:
                problems.append("bad_length")
                return None, None
            address, value = struct.unpack(">HH", pdu[1:5])
            if function == 5 and value not in (0x0000, 0xFF00):
                problems.append("bad_coil_value")
            return address, 1
        if function in (15, 16):
            if n < 6:
                problems.append("bad_length")
                return None, None
            address, quantity, byte_count = struct.unpack(">HHB", pdu[1:6])
            if function == 15:
                ok = 1 <= quantity <= 1968 and byte_count == (quantity + 7) // 8
            else:
                ok = 1 <= quantity <= 123 and byte_count == 2 * quantity
            if not ok:
                problems.append("bad_quantity")
            if n != 6 + byte_count:
                problems.append("bad_length")
            return address, quantity
        return None, None

    @staticmethod
    def _response_fields(function: int, pdu: bytes, problems: List[str]) -> Tuple[Optional[int], Optional[int]]:
        n = len(pdu)
        if function in (1, 2, 3, 4):
            if n < 2 or n != 2 + pdu[1]:
                problems.append("bad_length")
            return None, None
        if function in (5, 6):
            if n != 5:
                problems.append("bad_length")
                return None, None
            return struct.unpack(">H", pdu[1:3])[0], 1
        if function in (15, 16):
            if n != 5:
                problems.append("bad_length")
                return None, None
            address, quantity = struct.unpack(">HH", pdu[1:5])
            return address, quantity
        return None, None


# --------------------------------------------------------------------------
# Frame builders (used by the simulator and the tests)
# --------------------------------------------------------------------------
def build_adu(tid: int, unit: int, pdu: bytes) -> bytes:
    return struct.pack(">HHHB", tid & 0xFFFF, 0, len(pdu) + 1, unit) + pdu


def read_request(tid: int, unit: int, function: int, address: int, quantity: int) -> bytes:
    return build_adu(tid, unit, bytes([function]) + struct.pack(">HH", address, quantity))


def read_response(tid: int, unit: int, function: int, values: List[int]) -> bytes:
    if function in (3, 4):
        data = b"".join(struct.pack(">H", v & 0xFFFF) for v in values)
    else:
        data = bytes(v & 0xFF for v in values)
    return build_adu(tid, unit, bytes([function, len(data)]) + data)


def write_single_request(tid: int, unit: int, function: int, address: int, value: int) -> bytes:
    return build_adu(tid, unit, bytes([function]) + struct.pack(">HH", address, value & 0xFFFF))


write_single_response = write_single_request  # the device echoes the request


def write_multiple_request(tid: int, unit: int, address: int, values: List[int]) -> bytes:
    data = b"".join(struct.pack(">H", v & 0xFFFF) for v in values)
    pdu = bytes([16]) + struct.pack(">HHB", address, len(values), len(data)) + data
    return build_adu(tid, unit, pdu)


def write_multiple_response(tid: int, unit: int, address: int, quantity: int) -> bytes:
    return build_adu(tid, unit, bytes([16]) + struct.pack(">HH", address, quantity))


def exception_response(tid: int, unit: int, function: int, code: int) -> bytes:
    return build_adu(tid, unit, bytes([(function | 0x80) & 0xFF, code]))


def raw_request(tid: int, unit: int, function: int, data: bytes = b"") -> bytes:
    return build_adu(tid, unit, bytes([function & 0xFF]) + data)
