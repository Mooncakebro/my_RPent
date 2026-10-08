"""Observation adapter and lock-step environment used by the bridge agent.

Only data received from the evaluator websocket is retained.  Camera poses are
derived from the public ``cam_rel_poses`` observation when present; they are
reported in the robot-base frame, never as simulator/world state.
"""

from __future__ import annotations

import io
import queue
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np

from robots.behavior.schemas import ACTION_DIM, RAW_PROPRIO_SEGMENTS, validate_action_chunk

from .protocol import sanitize_action

try:  # imageio is an RPent behavior dependency, but keep imports lightweight.
    import imageio.v2 as _imageio
except ImportError:  # pragma: no cover - only used when observe() is called
    _imageio = None


_CAMERAS = ("head", "left_wrist", "right_wrist")
_SENSOR_SUFFIXES = {
    "head": ("zed_link:Camera:0", "head"),
    "left_wrist": ("left_realsense_link:Camera:0", "left_wrist"),
    "right_wrist": ("right_realsense_link:Camera:0", "right_wrist"),
}
_PROPRIO_SUFFIXES = {
    "arm_left_qpos": RAW_PROPRIO_SEGMENTS["left_arm"],
    "arm_right_qpos": RAW_PROPRIO_SEGMENTS["right_arm"],
    "gripper_left_qpos": RAW_PROPRIO_SEGMENTS["left_gripper"],
    "gripper_right_qpos": RAW_PROPRIO_SEGMENTS["right_gripper"],
    "trunk_qpos": RAW_PROPRIO_SEGMENTS["trunk"],
    "base_qvel": RAW_PROPRIO_SEGMENTS["base"],
}


def _first(value: Any) -> Any:
    array = np.asarray(value)
    return array[0] if array.ndim and array.shape[0] == 1 else array


def _find(mapping: Mapping[str, Any], *parts: str) -> Any | None:
    for key, value in mapping.items():
        text = str(key)
        if all(part in text for part in parts):
            return value
    return None


def _quat_matrix(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.size != 4:
        return np.eye(3, dtype=np.float32)
    # Omnigibson's public relative pose uses xyzw.  Normalize defensively.
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _pose_matrix(pose: Any) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float32).reshape(-1)
    matrix = np.eye(4, dtype=np.float32)
    if value.size >= 7 and np.isfinite(value[:7]).all():
        matrix[:3, :3] = _quat_matrix(value[3:7])
        matrix[:3, 3] = value[:3]
    return matrix


def _png_bytes(image: np.ndarray) -> bytes:
    if _imageio is None:
        raise RuntimeError("imageio is required for bridge.observe()")
    buffer = io.BytesIO()
    _imageio.imwrite(buffer, np.asarray(image)[..., :3], format="png")
    return buffer.getvalue()


def _raw_proprio(observation: Mapping[str, Any]) -> np.ndarray:
    direct = observation.get("states")
    if direct is not None:
        value = np.asarray(_first(direct), dtype=np.float32).reshape(-1)
        if value.size >= 256:
            return np.ascontiguousarray(value)
    result = np.zeros(256, dtype=np.float32)
    found = False
    for name, segment in _PROPRIO_SUFFIXES.items():
        value = _find(observation, name)
        if value is None:
            continue
        values = np.asarray(_first(value), dtype=np.float32).reshape(-1)
        width = (segment.stop or 0) - (segment.start or 0)
        result[segment] = np.resize(values, width)
        found = True
    if not found:
        raise ValueError("evaluator observation contains no R1Pro proprioception")
    return result


def adapt_observation(
    observation: Mapping[str, Any],
    *,
    task_language: str = "",
    camera_intrinsics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert flattened evaluator fields to RPent's public Behavior shape."""

    cameras: dict[str, dict[str, Any]] = {}
    rel_poses = _find(observation, "cam_rel_poses")
    rel_array = None if rel_poses is None else np.asarray(_first(rel_poses), dtype=np.float32).reshape(-1)
    default_intrinsics = {
        "head": np.array([[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "left_wrist": np.array([[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "right_wrist": np.array([[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    }
    # Evaluator's R1Pro YAML inserts camera roles left wrist, right wrist,
    # then head; Evaluator._preprocess_obs preserves that insertion order when
    # concatenating the public cam_rel_poses field.
    pose_index = {"left_wrist": 0, "right_wrist": 1, "head": 2}
    for camera in _CAMERAS:
        suffixes = _SENSOR_SUFFIXES[camera]
        rgb = None
        for suffix in suffixes:
            rgb = _find(observation, suffix, "rgb")
            if rgb is not None:
                break
        if rgb is None:
            raise ValueError(f"evaluator observation lacks {camera} RGB")
        image = np.asarray(_first(rgb))
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError(f"{camera} RGB must be [H,W,3]")
        depth = None
        for suffix in suffixes:
            depth = _find(observation, suffix, "depth_linear")
            if depth is not None:
                break
        depth_array = (
            np.asarray(_first(depth), dtype=np.float32).squeeze()
            if depth is not None
            else np.full(image.shape[:2], np.nan, dtype=np.float32)
        )
        index = pose_index[camera]
        if rel_array is not None and rel_array.size >= (index + 1) * 7:
            pose = _pose_matrix(rel_array[index * 7 : (index + 1) * 7])
        else:
            pose = np.eye(4, dtype=np.float32)
        cameras[camera] = {
            "rgb": np.ascontiguousarray(image[..., :3].astype(np.uint8, copy=False)),
            "depth": np.ascontiguousarray(depth_array),
            "intrinsic": np.asarray((camera_intrinsics or {}).get(camera, default_intrinsics[camera]), dtype=np.float32),
            "camera_to_world": pose,
        }
    return {
        "main_images": cameras["head"]["rgb"],
        "wrist_images": np.stack([cameras["left_wrist"]["rgb"], cameras["right_wrist"]["rgb"]]),
        "states": _raw_proprio(observation),
        "task_descriptions": task_language,
        "extra_view_images": None,
        "_bridge_cameras": cameras,
    }


class BridgeRendezvous:
    """Capacity-one queues joining the asyncio websocket and agent thread."""

    def __init__(self) -> None:
        self.observations: queue.Queue[Mapping[str, Any]] = queue.Queue(maxsize=1)
        self.actions: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self.current: dict[str, Any] | None = None
        self.closed = False

    def reset(self) -> None:
        with self._lock:
            self.closed = False
            self.current = None
            for slot in (self.observations, self.actions):
                while True:
                    try:
                        slot.get_nowait()
                    except queue.Empty:
                        break

    def put_observation(self, observation: Mapping[str, Any]) -> None:
        if self.closed:
            return
        try:
            self.observations.put(dict(observation), timeout=300.0)
        except queue.Full as exc:
            raise RuntimeError("agent did not consume evaluator observation") from exc

    def put_action(self, action: Any) -> None:
        if self.closed:
            return
        self.actions.put(sanitize_action(action), timeout=300.0)

    def close(self) -> None:
        self.closed = True
        # Wake an agent blocked in reset()/step() so an episode reset cannot
        # leave a daemon planner thread behind.
        try:
            self.observations.put_nowait({"_bridge_closed": True})
        except queue.Full:
            pass
        try:
            self.actions.put_nowait(np.zeros(ACTION_DIM, dtype=np.float32))
        except queue.Full:
            pass


class ShimBehaviorEnv:
    """Duck-typed Behavior env consumed by ``BehaviorPrimitives``."""

    def __init__(self, rendezvous: BridgeRendezvous, *, task_language: str = "") -> None:
        self.rendezvous = rendezvous
        self.task_language = task_language
        self.last_observation: dict[str, Any] | None = None
        self.last_info: dict[str, Any] = {}
        self.total_env_steps = 0
        self._gripper = {"left": 1.0, "right": 1.0}
        self._finished = False

    @property
    def official_success_latched(self) -> bool:
        return False

    @property
    def official_success_receipt(self) -> None:
        return None

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        raw = self.rendezvous.observations.get(timeout=300.0)
        if raw.get("_bridge_closed"):
            raise RuntimeError("bridge episode was reset")
        self.last_observation = adapt_observation(raw, task_language=self.task_language)
        self.total_env_steps = 0
        self.last_info = {"_bridge": True}
        return self.last_observation, self.last_info

    def current_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.last_observation is None:
            return self.reset()
        return self.last_observation, self.last_info

    def _hold_action_from_current_proprio(self) -> np.ndarray:
        if self.last_observation is None:
            return np.zeros(ACTION_DIM, dtype=np.float32)
        raw = np.asarray(self.last_observation["states"], dtype=np.float32)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[0:3] = 0
        action[3:7] = raw[RAW_PROPRIO_SEGMENTS["trunk"]]
        action[7:14] = raw[RAW_PROPRIO_SEGMENTS["left_arm"]]
        action[14] = self._gripper["left"]
        action[15:22] = raw[RAW_PROPRIO_SEGMENTS["right_arm"]]
        action[22] = self._gripper["right"]
        return sanitize_action(action)

    def _receive_after_action(self, action: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        safe = sanitize_action(action, current=self._hold_action_from_current_proprio())
        self.rendezvous.put_action(safe)
        raw = self.rendezvous.observations.get(timeout=600.0)
        if raw.get("_bridge_closed"):
            raise RuntimeError("bridge episode was reset")
        self.last_observation = adapt_observation(raw, task_language=self.task_language)
        self.total_env_steps += 1
        self.last_info = {"_bridge": True, "executed_steps": 1, "stop_reason": "lockstep"}
        return self.last_observation, self.last_info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, info = self._receive_after_action(action)
        return obs, 0.0, False, False, info

    def chunk_step(self, actions: Any, *, return_all_frames: bool = False) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        chunk = validate_action_chunk(actions)
        frames: list[dict[str, Any]] = []
        info: dict[str, Any] = {}
        for action in chunk:
            obs, _reward, _term, _trunc, info = self.step(action)
            if return_all_frames:
                frames.append(obs)
        return (frames if return_all_frames else self.last_observation), 0.0, False, False, info

    def observe(self, camera: str = "head", **_kwargs: Any) -> dict[str, Any]:
        observation, info = self.current_observation()
        cameras = observation.get("_bridge_cameras", {})
        camera = "head" if camera == "main" else camera
        frame = cameras.get(camera)
        if frame is None:
            raise ValueError(f"unknown camera {camera!r}")
        return {
            "status": "ok",
            "camera": camera,
            "frame_id": f"bridge-{self.total_env_steps}-{camera}",
            "step": self.total_env_steps,
            "_image_bytes": _png_bytes(frame["rgb"]),
            "_depth_image_bytes": _png_bytes(np.nan_to_num(frame["depth"], nan=0.0)[..., None].repeat(3, axis=-1)),
            "intrinsic": frame["intrinsic"],
            "camera_to_world": frame["camera_to_world"],
            "info": info,
        }

    def get_camera_meta(self, camera_name: str = "head", **_kwargs: Any) -> dict[str, Any]:
        observation, _ = self.current_observation()
        camera = "head" if camera_name == "main" else camera_name
        frame = observation["_bridge_cameras"][camera]
        return {"camera_name": camera, "rgb_shape": list(frame["rgb"].shape), "depth_available": bool(np.isfinite(frame["depth"]).any()), "intrinsic": frame["intrinsic"], "camera_to_world": frame["camera_to_world"]}

    def pixel_to_world(self, **kwargs: Any) -> dict[str, Any]:
        camera = kwargs.get("camera", "head")
        frame = self.current_observation()[0]["_bridge_cameras"][camera]
        u, v = int(kwargs["u"]), int(kwargs["v"])
        depth = frame["depth"]
        if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
            raise ValueError("pixel is outside the observed image")
        z = float(depth[v, u])
        if not np.isfinite(z) or z <= 0:
            return {"status": "failed", "primitive_success": False, "stop_reason": "invalid_depth"}
        k = frame["intrinsic"]
        optical = np.array([(u - k[0, 2]) * z / k[0, 0], -(v - k[1, 2]) * z / k[1, 1], -z, 1], dtype=np.float32)
        point = (frame["camera_to_world"] @ optical)[:3]
        return {"status": "ok", "primitive_success": True, "camera": camera, "world_xyz": point.tolist(), "depth_m": z, "frame_id": kwargs.get("frame_id")}

    def _unsupported(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return {"status": "failed", "name": name, "primitive_success": False, "stop_reason": "bridge_motion_unavailable", "error": "motion planning is not available from evaluator RGB/depth/proprio observations", "request": dict(kwargs)}

    def navigate_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._unsupported("navigate_to", **kwargs)

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._unsupported("move_to", **kwargs)

    def rotate_wrist(self, **kwargs: Any) -> dict[str, Any]:
        return self._unsupported("rotate_wrist", **kwargs)

    def _gripper_command(self, hand: str, command: float, name: str, **kwargs: Any) -> dict[str, Any]:
        if hand not in self._gripper:
            return self._unsupported(name, **kwargs)
        self._gripper[hand] = command
        action = self._hold_action_from_current_proprio()
        action[14 if hand == "left" else 22] = command
        start = self.total_env_steps
        info: dict[str, Any] = {}
        # Match RPent's existing gripper primitive duration while preserving
        # the evaluator's strict one-action/one-step lockstep.
        for _ in range(15):
            _obs, _reward, _terminated, _truncated, info = self.step(action)
        return {
            "status": "ok",
            "name": name,
            "primitive_success": True,
            "task_success": False,
            "hand": hand,
            "gripper_command": command,
            "executed_steps": self.total_env_steps - start,
            "stop_reason": "requested_actions_completed",
            "info": info,
            "_observation": self.last_observation,
        }

    def close_gripper(self, **kwargs: Any) -> dict[str, Any]:
        hand = kwargs.get("hand")
        return self._gripper_command(str(hand), -1.0, "close", **kwargs)

    def open_gripper(self, **kwargs: Any) -> dict[str, Any]:
        hand = kwargs.get("hand")
        return self._gripper_command(str(hand), 1.0, "open", **kwargs)

    def press(self, **kwargs: Any) -> dict[str, Any]:
        return self._unsupported("press", **kwargs)

    def close_transport(self) -> None:
        self.rendezvous.close()


__all__ = ["BridgeRendezvous", "ShimBehaviorEnv", "adapt_observation"]
