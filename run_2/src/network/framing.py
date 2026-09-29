"""Length-prefixed protobuf framing for TCP transport.

Every protobuf Envelope sent over TCP is prefixed with a 4-byte big-endian
unsigned integer indicating the payload length.  This module handles both
the encoding side (``encode_message``) and the decoding side (``FrameReader``).

The FrameReader is a streaming state machine: the transport server feeds
arbitrary-sized chunks from ``sock_recv`` into ``feed()``, and the reader
accumulates data, extracts complete frames, parses them into Envelope
messages, and returns them.  This correctly handles the case where a single
``sock_recv`` contains a partial message, multiple messages, or any mix.

With Stage 3 tc/netem rules introducing real packet loss and latency, there
is an increased chance of receiving corrupted or truncated data.  The reader
validates that ``ParseFromString`` consumed the expected number of bytes and
that the resulting Envelope has a non-empty payload, logging warnings for
any anomalies.
"""

from __future__ import annotations

import logging
import struct

from src.proto_gen import federation_pb2

logger = logging.getLogger(__name__)

HEADER_SIZE = 4  # 4-byte big-endian uint32 length prefix
MAX_MESSAGE_SIZE = 64 * 1024 * 1024  # 64 MB max message size


def encode_message(envelope: federation_pb2.Envelope) -> bytes:
    """Serialize an Envelope to a length-prefixed bytes buffer.

    The format is: [4-byte BE length][serialized protobuf payload].
    The length prefix counts only the payload bytes, not itself.
    """
    payload = envelope.SerializeToString()
    header = struct.pack(">I", len(payload))
    return header + payload


class FrameReader:
    """Streaming state machine that accumulates TCP data and yields Envelopes.

    Handles partial reads correctly since ``loop.sock_recv`` may return
    fewer bytes than a complete message.  Also validates parsed messages
    to catch data corruption early rather than passing malformed envelopes
    silently to the handler.
    """

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[federation_pb2.Envelope]:
        """Feed raw bytes from sock_recv.  Returns list of complete messages.

        Each call may return zero, one, or many messages depending on how
        much data was buffered and how many complete frames are available.
        Malformed messages (where protobuf parsing fails or produces an
        empty envelope) are logged and skipped rather than propagated.
        """
        self._buffer.extend(data)
        messages = []
        while True:
            if len(self._buffer) < HEADER_SIZE:
                break
            payload_len = struct.unpack(">I", self._buffer[:HEADER_SIZE])[0]
            if payload_len > MAX_MESSAGE_SIZE:
                raise ValueError(f"Message too large: {payload_len} bytes")
            total_needed = HEADER_SIZE + payload_len
            if len(self._buffer) < total_needed:
                break
            # Extract complete frame
            payload_bytes = bytes(self._buffer[HEADER_SIZE:total_needed])
            del self._buffer[:total_needed]
            # Parse and validate
            envelope = federation_pb2.Envelope()
            bytes_consumed = envelope.ParseFromString(payload_bytes)
            if bytes_consumed != len(payload_bytes):
                logger.warning(
                    f"Protobuf parse consumed {bytes_consumed}/{len(payload_bytes)} "
                    f"bytes — possible data corruption, skipping message"
                )
                continue
            if not envelope.WhichOneof("payload"):
                logger.warning("Received envelope with no payload, skipping")
                continue
            messages.append(envelope)
        return messages

    def has_partial_frame(self) -> bool:
        """True when bytes of an incomplete frame are still buffered.

        The transport server uses this to attribute first-byte arrival
        stamps: leftover bytes after a ``feed`` belong to the *next* frame,
        so that frame's first byte arrived in the read just processed.
        """
        return bool(self._buffer)

    def reset(self):
        """Discard any buffered partial data."""
        self._buffer.clear()
