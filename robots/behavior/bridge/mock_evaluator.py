"""Offline evaluator client for exercising the bridge without OmniGibson."""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from .protocol import ACTION_DIM, MsgpackCodec


def synthetic_observation(step: int = 0, *, height: int = 32, width: int = 32) -> dict[str, Any]:
    """Create a valid flattened RGB-D/proprio evaluator observation."""

    rgb = np.full((height, width, 3), step % 255, dtype=np.uint8)
    depth = np.full((height, width), 1.0, dtype=np.float32)
    return {
        "robot_r1:zed_link:Camera:0::rgb": rgb,
        "robot_r1:left_realsense_link:Camera:0::rgb": rgb.copy(),
        "robot_r1:right_realsense_link:Camera:0::rgb": rgb.copy(),
        "robot_r1:zed_link:Camera:0::depth_linear": depth,
        "robot_r1:left_realsense_link:Camera:0::depth_linear": depth.copy(),
        "robot_r1:right_realsense_link:Camera:0::depth_linear": depth.copy(),
        "robot_r1::arm_left_qpos": np.zeros(7, dtype=np.float32),
        "robot_r1::arm_right_qpos": np.zeros(7, dtype=np.float32),
        "robot_r1::gripper_left_qpos": np.zeros(2, dtype=np.float32),
        "robot_r1::gripper_right_qpos": np.zeros(2, dtype=np.float32),
        "robot_r1::trunk_qpos": np.zeros(4, dtype=np.float32),
        "robot_r1::base_qvel": np.zeros(3, dtype=np.float32),
        "robot_r1::cam_rel_poses": np.tile(np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), 3),
        "task_id": np.asarray([0], dtype=np.int64),
    }


class MockEvaluator:
    """Synchronous protocol client matching ``WebsocketClientPolicy``."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, *, reconnect: bool = True) -> None:
        self.host = host
        self.port = int(port)
        self.reconnect = reconnect
        self.codec = MsgpackCodec()
        self._connection: Any = None
        self.metadata: dict[str, Any] = {}
        self.steps = 0

    def _connect(self) -> None:
        try:
            import requests
            import websockets.sync.client
        except ImportError as exc:  # pragma: no cover - dependency dependent
            raise RuntimeError("requests and websockets are required for mock evaluator") from exc
        health_url = f"http://{self.host}:{self.port}/healthz"
        for _ in range(120):
            try:
                if requests.get(health_url, timeout=2).ok:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise TimeoutError(f"bridge health check did not pass: {health_url}")
        self._connection = websockets.sync.client.connect(
            f"ws://{self.host}:{self.port}",
            compression=None,
            max_size=None,
            ping_interval=60,
            ping_timeout=300,
        )
        metadata = self.codec.unpack(self._connection.recv())
        if not isinstance(metadata, Mapping):
            raise RuntimeError("bridge metadata must be a mapping")
        self.metadata = dict(metadata)

    def reset(self) -> None:
        if self._connection is None:
            self._connect()
        self._connection.send(self.codec.pack({"reset": True}))
        self.steps = 0

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self._connection is None:
            self._connect()
        payload = self.codec.pack(dict(observation))
        for attempt in range(3):
            try:
                self._connection.send(payload)
                response = self.codec.unpack(self._connection.recv())
                if isinstance(response, Mapping) and "action" in response:
                    action = np.asarray(response["action"], dtype=np.float32)
                    if action.shape != (ACTION_DIM,):
                        raise RuntimeError(f"invalid action shape: {action.shape}")
                    self.steps += 1
                    return action
                if attempt == 2:
                    raise RuntimeError(f"bridge response missing action: {response!r}")
            except Exception:
                if not self.reconnect or attempt == 2:
                    raise
                self.close()
                self._connect()
        raise AssertionError("unreachable")

    def run(self, steps: int = 3) -> list[np.ndarray]:
        self.reset()
        actions = []
        for step in range(int(steps)):
            actions.append(self.act(synthetic_observation(step)))
        return actions

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    client = MockEvaluator(args.host, args.port)
    try:
        actions = client.run(args.steps)
        print(f"completed {len(actions)} lockstep steps; action shape={actions[-1].shape if actions else None}")
    finally:
        client.close()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["MockEvaluator", "synthetic_observation"]
