"""Passive websocket bridge for official BEHAVIOR evaluation."""

from .protocol import ACTION_DIM, BridgeProtocolError, MsgpackCodec

__all__ = ["ACTION_DIM", "BridgeProtocolError", "MsgpackCodec"]
