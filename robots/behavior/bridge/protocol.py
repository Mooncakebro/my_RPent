"""Wire protocol and websocket server for the BEHAVIOR evaluator bridge.

The msgpack representation intentionally mirrors BEHAVIOR-1K's
``eval.utils.network_utils``.  In particular, NumPy arrays use a msgpack map
with ``__ndarray__``, ``data``, ``dtype`` and ``shape`` fields.  No simulator
objects are imported here; this module is safe to import in the RPent venv.

The codec is adapted from BEHAVIOR-1K's Apache-2.0-compatible evaluator
implementation, which in turn credits Physical Intelligence/openpi and
lebedov/msgpack-numpy.  The adaptation retains their protocol field names and
is intentionally kept local so the bridge does not import simulator modules.
"""

from __future__ import annotations

import asyncio
import functools
import http
import inspect
import logging
import traceback
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import msgpack
import numpy as np

ACTION_DIM = 23
LOGGER = logging.getLogger(__name__)


class BridgeProtocolError(RuntimeError):
    """Raised when an evaluator message or policy action is invalid."""


def _pack_data(value: Any) -> Any:
    """Encode NumPy values using the official evaluator extension format."""

    # Torch is deliberately optional in the bridge environment.  Import it
    # only when a tensor-like value is actually presented to the codec.
    if value.__class__.__module__.split(".", 1)[0] == "torch":
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise TypeError(f"unsupported NumPy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_data(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    if value.get(b"__ndarray__") or value.get("__ndarray__"):
        get = value.get
        raw = get(b"data", get("data"))
        dtype = get(b"dtype", get("dtype"))
        shape = get(b"shape", get("shape"))
        try:
            return np.ndarray(buffer=raw, dtype=np.dtype(dtype), shape=tuple(shape))
        except (TypeError, ValueError) as exc:
            raise BridgeProtocolError("invalid ndarray payload") from exc
    if value.get(b"__npgeneric__") or value.get("__npgeneric__"):
        get = value.get
        return np.dtype(get(b"dtype", get("dtype"))).type(get(b"data", get("data")))
    return value


Packer = functools.partial(msgpack.Packer, default=_pack_data)
packb = functools.partial(msgpack.packb, default=_pack_data)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=_unpack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_data)


class MsgpackCodec:
    """Small state-free codec matching the official evaluator's codec."""

    def __init__(self) -> None:
        self.packer = Packer()

    def pack(self, value: Any) -> bytes:
        return self.packer.pack(value)

    @staticmethod
    def unpack(payload: bytes | bytearray | memoryview) -> Any:
        return unpackb(payload, strict_map_key=False)


def sanitize_action(action: Any, *, current: Any | None = None) -> np.ndarray:
    """Return a finite contiguous float32 action with the official shape.

    Invalid actions are replaced by the current/last action when available,
    otherwise zeros.  This makes the bridge fail safe without ever sending a
    malformed message to the evaluator.
    """

    try:
        if action.__class__.__module__.split(".", 1)[0] == "torch":
            action = action.detach().cpu().numpy()
        array = np.asarray(action, dtype=np.float32)
    except (TypeError, ValueError):
        array = np.empty((0,), dtype=np.float32)
    if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
        if current is not None:
            try:
                fallback = np.asarray(current, dtype=np.float32)
                if fallback.shape == (ACTION_DIM,) and np.isfinite(fallback).all():
                    return np.ascontiguousarray(fallback)
            except (TypeError, ValueError):
                pass
        return np.zeros(ACTION_DIM, dtype=np.float32)
    return np.ascontiguousarray(array)


def _health_check(connection: Any, request: Any) -> Any:
    """Handle the evaluator's ``GET /healthz`` probe."""

    path = getattr(request, "path", None)
    if path != "/healthz":
        return None
    respond = getattr(connection, "respond", None)
    if callable(respond):
        return respond(http.HTTPStatus.OK, "OK\n")
    return http.HTTPStatus.OK, [("Content-Type", "text/plain")], b"OK\n"


class WebsocketBridgeServer:
    """Serve a policy callback over the official BEHAVIOR websocket protocol.

    ``policy`` may be synchronous or asynchronous.  Synchronous calls run in
    a worker thread so the asyncio loop continues to service websocket
    ping/pong frames while an LLM turn is in progress.
    """

    def __init__(
        self,
        policy: Any,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Mapping[str, Any] | None = None,
        ping_interval: float = 60.0,
        ping_timeout: float = 300.0,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = int(port)
        self.metadata = dict(metadata or {})
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.codec = MsgpackCodec()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        try:
            import websockets.asyncio.server as websocket_server
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "websockets is required to run the bridge server; install "
                "the RPent behavior bridge dependencies"
            ) from exc
        LOGGER.info("Starting BEHAVIOR bridge on %s:%s", self.host, self.port)
        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
        ) as server:
            await server.serve_forever()

    async def _call_policy(self, observation: Mapping[str, Any]) -> Any:
        callback = getattr(self.policy, "act", self.policy)
        if not callable(callback):
            raise BridgeProtocolError("policy must expose callable act(obs)")
        payload = deepcopy(dict(observation))
        if inspect.iscoroutinefunction(callback):
            return await callback(payload)
        # A synchronous planner can take minutes.  Do not block the event loop,
        # otherwise websocket control frames cannot be answered.
        return await asyncio.to_thread(callback, payload)

    async def _handler(self, websocket: Any) -> None:
        remote = getattr(websocket, "remote_address", None)
        LOGGER.info("Evaluator connection opened: %s", remote)
        await websocket.send(self.codec.pack(self.metadata))
        try:
            while True:
                raw = await websocket.recv()
                if isinstance(raw, str):
                    raise BridgeProtocolError("evaluator sent a text websocket frame")
                message = self.codec.unpack(raw)
                if not isinstance(message, Mapping):
                    raise BridgeProtocolError("evaluator message must be a mapping")
                if message.get("reset") or message.get(b"reset"):
                    reset = getattr(self.policy, "reset", None)
                    if callable(reset):
                        result = reset()
                        if asyncio.iscoroutine(result):
                            await result
                    # The official client does not await a reset response.
                    continue
                action = sanitize_action(await self._call_policy(message))
                await websocket.send(self.codec.pack({"action": action}))
        except Exception as exc:
            # Connection closure is normal and should not be reported as a
            # policy failure.  Avoid importing a concrete websockets exception
            # at module import time.
            if exc.__class__.__name__ not in {
                "ConnectionClosed",
                "ConnectionClosedOK",
                "ConnectionClosedError",
            }:
                LOGGER.error("Bridge connection error: %s\n%s", remote, traceback.format_exc())
                try:
                    await websocket.close(code=1011, reason="bridge policy error")
                except Exception:
                    pass
        finally:
            LOGGER.info("Evaluator connection closed: %s", remote)


__all__ = [
    "ACTION_DIM",
    "BridgeProtocolError",
    "MsgpackCodec",
    "Packer",
    "Unpacker",
    "WebsocketBridgeServer",
    "WebsocketPolicyServer",
    "packb",
    "sanitize_action",
    "unpackb",
]

# Naming used by BEHAVIOR-1K's evaluator-side implementation.  Keeping this
# alias makes the bridge a drop-in policy server for local launch scripts.
WebsocketPolicyServer = WebsocketBridgeServer
