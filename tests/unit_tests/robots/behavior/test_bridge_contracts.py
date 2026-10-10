from __future__ import annotations

import threading

import numpy as np

from robots.behavior.bridge.mock_evaluator import synthetic_observation
from robots.behavior.bridge.protocol import ACTION_DIM, MsgpackCodec, sanitize_action
from robots.behavior.bridge.server import AgentSessionPolicy
from robots.behavior.bridge.shim_env import BridgeRendezvous, ShimBehaviorEnv, adapt_observation
from robots.behavior.prompts.bridge_system import system_prompt
from rpent.prompt.utils import format_prompt


def test_float_depth_png_encodes_without_pillow_crash() -> None:
    """Depth is float32; the PNG encoder must clip+cast, not raise the Pillow
    'Cannot handle this data type' error seen in the first bridge smoke."""

    from robots.behavior.bridge.shim_env import _png_bytes

    depth = np.full((1, 1), 1.5, dtype=np.float32)
    payload = _png_bytes(depth[..., None].repeat(3, axis=-1))
    assert isinstance(payload, bytes) and payload[:4] == b"\x89PNG"
    # large floats must not wrap/overflow
    payload2 = _png_bytes(np.full((16, 16, 3), 300.0, dtype=np.float32))
    assert payload2[:4] == b"\x89PNG"


def test_flattened_proprio_maps_to_raw_segments() -> None:
    """The official evaluator sends ONE ``<robot>::proprio`` vector; the shim
    must scatter its PROPRIOCEPTION_INDICES slices onto the 256-dim raw layout.
    """

    from robots.behavior.schemas import RAW_PROPRIO_SEGMENTS

    observation = adapt_observation(synthetic_observation())
    states = observation["states"]
    assert states.shape == (256,)
    assert np.allclose(states[RAW_PROPRIO_SEGMENTS["left_arm"]], np.arange(7) + 10)
    assert np.allclose(states[RAW_PROPRIO_SEGMENTS["right_arm"]], np.arange(7) + 20)
    assert np.allclose(states[RAW_PROPRIO_SEGMENTS["left_gripper"]], [0.5, 0.5])
    assert np.allclose(states[RAW_PROPRIO_SEGMENTS["right_gripper"]], [0.5, 0.5])
    assert np.allclose(states[RAW_PROPRIO_SEGMENTS["trunk"]], [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(states[RAW_PROPRIO_SEGMENTS["base"]], [5.0, 6.0, 7.0])


def test_double_reset_during_initialization_spawns_one_agent() -> None:
    import time as _time

    from robots.behavior.bridge.server import AgentSessionPolicy

    calls: list = []
    started = threading.Event()
    release = threading.Event()

    def slow_factory(env, **kwargs):
        calls.append(env)
        started.set()
        release.wait(timeout=5)
        return None

    policy = AgentSessionPolicy(agent_factory=slow_factory, task_language="test")
    policy.reset()
    assert started.wait(timeout=2)
    policy.reset()  # redundant reset while the factory is still initializing
    release.set()
    deadline = _time.time() + 5
    while _time.time() < deadline and len(calls) < 2:
        _time.sleep(0.05)
    assert len(calls) == 1


def test_msgpack_round_trip_nested_numpy_tree() -> None:
    value = {"rgb": np.zeros((2, 3, 3), dtype=np.uint8), "nested": {"x": np.arange(4, dtype=np.float32)}}
    result = MsgpackCodec.unpack(MsgpackCodec().pack(value))
    assert np.array_equal(result["rgb"], value["rgb"])
    assert np.array_equal(result["nested"]["x"], value["nested"]["x"])


def test_observation_adapter_and_capacity_one_lockstep() -> None:
    rendezvous = BridgeRendezvous()
    env = ShimBehaviorEnv(rendezvous, task_language="test")
    rendezvous.put_observation(synthetic_observation())
    observation, _ = env.reset()
    assert observation["main_images"].dtype == np.uint8
    assert observation["wrist_images"].shape[0] == 2
    assert observation["states"].shape == (256,)

    result: list[tuple] = []
    worker = threading.Thread(target=lambda: result.append(env.step(np.ones(ACTION_DIM))))
    worker.start()
    action = rendezvous.actions.get(timeout=1)
    assert action.shape == (ACTION_DIM,)
    assert action.dtype == np.float32
    rendezvous.put_observation(synthetic_observation(1))
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert result[0][0]["main_images"].shape == observation["main_images"].shape


def test_agent_session_returns_hold_after_agent_finishes() -> None:
    def factory(env):
        def run():
            env.reset()
            env.step(np.arange(ACTION_DIM, dtype=np.float32))

        return run

    policy = AgentSessionPolicy(agent_factory=factory)
    policy.reset()
    first = policy.act(synthetic_observation())
    second = policy.act(synthetic_observation(1))
    assert first.shape == (ACTION_DIM,)
    assert np.array_equal(first, np.arange(ACTION_DIM, dtype=np.float32))
    assert second.shape == (ACTION_DIM,)
    assert np.isfinite(second).all()


def test_sanitize_action_falls_back_to_current() -> None:
    current = np.ones(ACTION_DIM, dtype=np.float32)
    result = sanitize_action([float("nan")], current=current)
    assert np.array_equal(result, current)


def test_bridge_prompt_restricts_unavailable_motion_tools() -> None:
    prompt = format_prompt(
        system_prompt(),
        variables={
            "mode": "bridge",
            "task_name": "turning_on_radio",
            "task_instruction": "Turn on the radio",
            "public_seed": 0,
            "recipe_tag": "bridge",
            "output_dir": "/tmp",
            "max_episode_steps": 10,
            "wall_clock_seconds": 10,
            "memory_dir": "/tmp/memory",
            "memory_profile": "hf",
            "memory_inbox": "/tmp/memory/inbox",
            "public_capabilities": ["observe"],
        },
    )
    assert "navigate_to" in prompt
    assert "unavailable" in prompt
    assert "pi0_nav_pick" in prompt
