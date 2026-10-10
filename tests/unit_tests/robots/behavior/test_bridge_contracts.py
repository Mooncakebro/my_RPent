from __future__ import annotations

import threading

import numpy as np

from robots.behavior.bridge.mock_evaluator import synthetic_observation
from robots.behavior.bridge.protocol import ACTION_DIM, MsgpackCodec, sanitize_action
from robots.behavior.bridge.server import AgentSessionPolicy
from robots.behavior.bridge.shim_env import BridgeRendezvous, ShimBehaviorEnv, adapt_observation
from robots.behavior.prompts.bridge_system import system_prompt
from rpent.prompt.utils import format_prompt


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
