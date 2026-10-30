"""Protocol-independent representation of one parsed application message."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

REQUEST = "request"  # client -> server
RESPONSE = "response"  # server -> client


@dataclass
class Message:
    """One Modbus/TCP ADU or one MQTT control packet.

    ``client`` and ``server`` are IP addresses. The *client* is the side that
    opened the connection to the well-known server port, whichever way this
    particular message travelled. ``direction`` says which way it went.
    Fields that do not apply to a protocol stay ``None``.
    """

    ts: float
    proto: str  # "modbus" | "mqtt"
    client: str
    server: str
    direction: str  # REQUEST | RESPONSE
    kind: str  # e.g. "fc3", "fc3_exc", "PUBLISH", "invalid"
    size: int  # bytes on the wire for this message
    valid: bool = True
    problems: Tuple[str, ...] = ()

    # Modbus
    unit_id: Optional[int] = None
    function: Optional[int] = None
    address: Optional[int] = None
    quantity: Optional[int] = None
    exception: Optional[int] = None

    # MQTT
    topic: Optional[str] = None
    qos: Optional[int] = None
    retain: bool = False
    payload_len: Optional[int] = None
    payload_entropy: Optional[float] = None
    client_id: Optional[str] = None
    filters: Tuple[str, ...] = ()
