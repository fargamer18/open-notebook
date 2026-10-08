"""Versioned JSON-lines protocol between LiteNotebook and its embedding worker.

The worker is a child process talking over its own stdin/stdout pipes, so no
other process can connect to it. Every line is one JSON object; nothing is
ever unpickled.

Requests (parent → worker)::

    {"v": 1, "id": 7, "op": "hello"}
    {"v": 1, "id": 8, "op": "embed", "texts": ["…", "…"]}
    {"v": 1, "id": 9, "op": "ping"}
    {"v": 1, "id": 10, "op": "shutdown"}

Responses (worker → parent) carry the same id::

    {"v": 1, "id": 8, "ok": true, "result": {"shape": [2, 384],
        "dtype": "<f4", "data": "<base64 of little-endian float32>"}}
    {"v": 1, "id": 8, "ok": false,
        "error": {"type": "model_unavailable", "message": "…"}}

Vectors travel as base64 float32 (a 64 × 384 batch is ~130 kB) and are
re-validated by the parent: count, dtype, dimension, finiteness and L2 norm.
This module is lightweight: it never imports torch or sentence-transformers.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import numpy.typing as npt

from litenotebook.config import EMBED_DIM

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 64 * 1024 * 1024
MAX_TEXTS_PER_REQUEST = 256
VECTOR_DTYPE = "<f4"
OPS = frozenset({"hello", "embed", "ping", "shutdown"})

# Structured error types the worker may report.
ERR_MODEL_UNAVAILABLE = "model_unavailable"
ERR_EMBEDDING = "embedding_error"
ERR_BAD_REQUEST = "bad_request"
ERR_INTERNAL = "internal_error"


class ProtocolError(ValueError):
    """A message that does not follow the protocol."""


def dumps(message: dict[str, Any]) -> bytes:
    return (
        json.dumps(message, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
    )


def loads(line: bytes) -> dict[str, Any]:
    if len(line) > MAX_LINE_BYTES:
        raise ProtocolError("message too large")
    try:
        message = json.loads(line)
    except (ValueError, UnicodeDecodeError) as e:
        raise ProtocolError("not a JSON message") from e
    if not isinstance(message, dict):
        raise ProtocolError("message is not an object")
    if message.get("v") != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version {message.get('v')!r} is not supported")
    if not isinstance(message.get("id"), int):
        raise ProtocolError("message has no integer id")
    return message


def request(request_id: int, op: str, **fields: Any) -> dict[str, Any]:
    assert op in OPS
    return {"v": PROTOCOL_VERSION, "id": request_id, "op": op, **fields}


def ok(request_id: int, result: dict[str, Any]) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "id": request_id, "ok": True, "result": result}


def error(request_id: int, kind: str, message: str) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "id": request_id,
        "ok": False,
        "error": {"type": kind, "message": message[:2000]},
    }


def encode_vectors(vectors: npt.NDArray[np.float32]) -> dict[str, Any]:
    array = np.ascontiguousarray(vectors, dtype=VECTOR_DTYPE)
    return {
        "shape": list(array.shape),
        "dtype": VECTOR_DTYPE,
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def decode_vectors(
    result: Any, expected_rows: int, dim: int = EMBED_DIM
) -> npt.NDArray[np.float32]:
    """Strictly decode a worker's vectors; never trust the shape it claims."""
    if not isinstance(result, dict):
        raise ProtocolError("embedding result is not an object")
    shape, dtype, data = result.get("shape"), result.get("dtype"), result.get("data")
    if dtype != VECTOR_DTYPE:
        raise ProtocolError(f"unexpected vector dtype {dtype!r}")
    if shape != [expected_rows, dim]:
        raise ProtocolError(
            f"expected {expected_rows} vectors of {dim} dims, got shape {shape!r}"
        )
    if not isinstance(data, str):
        raise ProtocolError("vector data is not a string")
    try:
        raw = base64.b64decode(data, validate=True)
    except ValueError as e:
        raise ProtocolError("vector data is not base64") from e
    if len(raw) != expected_rows * dim * 4:
        raise ProtocolError("vector data has the wrong length")
    array = np.frombuffer(raw, dtype=VECTOR_DTYPE).reshape(expected_rows, dim)
    return array.astype(np.float32)
