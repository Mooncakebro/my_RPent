from __future__ import annotations

import json
import threading

import numpy as np

from robots.behavior.bridge.mock_evaluator import synthetic_observation
from robots.behavior.bridge.protocol import ACTION_DIM, MsgpackCodec, sanitize_action
from robots.behavior.bridge.server import AgentSessionPolicy
from robots.behavior.bridge.shim_env import (
    BridgeRendezvous,
    ShimBehaviorEnv,
    adapt_observation,
)
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


def test_policy_checkpoint_measured_binding_for_unregistered_dir(tmp_path) -> None:
    """Unknown checkpoint dirs are measured-and-bound, not rejected; the
    PT50 profile still verifies its pins via the directory-name alias."""

    import json

    from robots.behavior import policy_checkpoint as pc

    files = {
        "model.safetensors": b"fake-weights",
        "config.json": json.dumps({"action_dim": 32}).encode(),
        "assets/behavior-1k/2025-challenge-demos/norm_stats.json": json.dumps({
            "norm_stats": {"state": {"mean": [], "q01": [], "q99": [], "std": []}}
        }).encode(),
    }
    for rel, data in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    binding = pc.validate_policy_checkpoint(tmp_path)
    assert binding.profile_id == tmp_path.name
    assert {item.relative_path for item in binding.files} == set(files)
    # a pinned, registered profile still rejects a mismatched file set
    alias_dir = tmp_path / "RLinf-Pi05-BEHAVIOR-1K-PT50-CS32"
    alias_dir.mkdir()
    (alias_dir / "model.safetensors").write_bytes(b"wrong")
    try:
        pc.validate_policy_checkpoint(alias_dir)
    except pc.PolicyCheckpointError:
        pass
    else:  # pragma: no cover
        raise AssertionError("pinned profile must reject mismatched weights")


def test_abort_wakes_agent_blocked_in_step() -> None:
    """When the evaluator closes the connection, abort() must wake a planner
    blocked waiting for the next observation so it can clean up and finish."""

    import time

    from robots.behavior.bridge.server import AgentSessionPolicy

    def blocking_agent(env):
        env.reset()  # consumes the observation we feed below
        env.step(np.zeros(23, dtype=np.float32))  # blocks for the next obs
        raise AssertionError("should not complete normally")  # pragma: no cover

    policy = AgentSessionPolicy(agent_factory=blocking_agent, task_language="test")
    policy.reset()
    assert policy.rendezvous is not None
    policy.rendezvous.put_observation(synthetic_observation())
    deadline = time.time() + 5
    while time.time() < deadline and policy.rendezvous.observations.qsize() == 0:
        time.sleep(0.02)
    policy.abort()
    deadline = time.time() + 5
    while time.time() < deadline and not policy._done.is_set():
        time.sleep(0.05)
    assert policy._done.is_set()
    assert policy._error is not None
    assert "bridge episode was reset" in str(policy._error)


def test_stale_generation_does_not_mark_new_agent_done() -> None:
    """A thread from a previous reset() must not set _done for the new agent."""

    import time

    from robots.behavior.bridge.server import AgentSessionPolicy

    class _BlockingAgent:
        def __init__(self, gate):
            self.gate = gate

        def run(self):
            self.gate.wait(timeout=5)

    gate_first = threading.Event()
    gate_second = threading.Event()
    next_gate = [gate_first]

    def factory(env, **kwargs):
        return _BlockingAgent(next_gate[0])

    policy = AgentSessionPolicy(agent_factory=factory, task_language="test")
    policy.reset()
    deadline = time.time() + 5
    while time.time() < deadline and policy._agent is None:
        time.sleep(0.02)
    assert policy._agent is not None  # agent 1 is running (blocked in run())
    first_agent = policy._agent
    next_gate[0] = gate_second
    policy.reset()  # retire generation 1 while agent 1 is still running
    deadline = time.time() + 5
    while time.time() < deadline and policy._agent is first_agent:
        time.sleep(0.02)
    assert policy._agent is not first_agent  # agent 2 spawned
    gate_first.set()  # old thread finishes; must NOT set _done for gen 2
    deadline = time.time() + 5
    while time.time() < deadline:
        if not policy._done.is_set():
            break
        time.sleep(0.02)
    assert not policy._done.is_set()
    gate_second.set()
    deadline = time.time() + 5
    while time.time() < deadline and not policy._done.is_set():
        time.sleep(0.05)
    assert policy._done.is_set()


def test_gripper_commands_accept_visual_hand_check(monkeypatch) -> None:
    """The bridge tool schema includes visual_hand_check; it must not collide
    with the positional ``hand`` argument used by the shim."""

    rendezvous = BridgeRendezvous()
    env = ShimBehaviorEnv(rendezvous, task_language="test")
    env.last_observation = adapt_observation(synthetic_observation())
    monkeypatch.setattr(
        env,
        "step",
        lambda action: (env.last_observation, 0.0, False, False, {}),
    )
    check = {
        "assessment": "selected_hand_visually_confirmed",
        "camera": "head",
        "frame_id": "bridge-0-head",
        "selected_hand": "left",
    }
    opened = env.open_gripper(hand="left", visual_hand_check=check)
    assert opened["status"] == "ok"
    assert opened["gripper_command"] == 1.0
    closed = env.close_gripper(hand="right", visual_hand_check=check)
    assert closed["status"] == "ok"
    assert closed["gripper_command"] == -1.0


def test_double_reset_after_running_agent_spawns_one_replacement() -> None:
    """The evaluator resets twice per instance. After an agent has run, the
    first reset replaces it; the second must not spawn a second VLA sidecar."""

    import time as _time

    from robots.behavior.bridge.server import AgentSessionPolicy

    calls: list = []
    second_started = threading.Event()
    release = threading.Event()
    agents: list = []

    class BlockingAgent:
        def __init__(self) -> None:
            self.stop = threading.Event()

        def run(self) -> None:
            self.stop.wait(timeout=10)

        def close(self) -> None:
            self.stop.set()

    def factory(env, **kwargs):
        calls.append(env)
        agent = BlockingAgent()
        agents.append(agent)
        if len(calls) == 1:
            return agent
        second_started.set()
        release.wait(timeout=5)
        return agent

    policy = AgentSessionPolicy(agent_factory=factory, task_language="test")
    policy.reset()
    deadline = _time.time() + 5
    while _time.time() < deadline and policy._agent is None:
        _time.sleep(0.02)
    assert policy._agent is not None

    policy.reset()  # new instance: retire the running agent
    assert second_started.wait(timeout=2)
    policy.reset()  # duplicate reset while the replacement is initializing
    release.set()
    deadline = _time.time() + 5
    while _time.time() < deadline and len(calls) < 2:
        _time.sleep(0.02)
    _time.sleep(0.2)
    assert len(calls) == 2
    for agent in agents:
        agent.stop.set()


def test_load_task_memory_excerpt_collects_published_files(tmp_path) -> None:
    from robots.behavior.bridge.agent import load_task_memory_excerpt

    (tmp_path / "task-specific").mkdir()
    (tmp_path / "task-specific" / "turning_on_radio_s0_recipe.jsonl").write_text(
        '{"action": "pi0_nav_pick", "instruction": "press red button"}\n'
        '{"action": "press", "hand": "right"}\n'
    )
    # A large audit file must NOT crowd out the recipe: recipes load first.
    (tmp_path / "task-specific" / "turning_on_radio_s0.json").write_text(
        "x" * 100_000
    )
    (tmp_path / "MEMORY.md").write_text("# Global notes\nradio is red\n")
    (tmp_path / "task-specific" / "picking_up_trash_recipe.jsonl").write_text(
        '{"steps": ["bin"]}\n'
    )
    excerpt = load_task_memory_excerpt(tmp_path, "turning_on_radio")
    assert "press red button" in excerpt  # recipe survives a huge audit file
    assert "radio is red" in excerpt
    assert "bin" not in excerpt  # other-task recipes are excluded
    # Simulator-only steps stay visible as context but are not presented as
    # bridge-callable actions.
    assert '"_standalone_only_step"' in excerpt
    assert '"bridge_substitute": "pi0_nav_pick with a precise instruction"' in excerpt
    recipe_section = excerpt.split("--- task-specific/turning_on_radio_s0_recipe.jsonl ---")[1]
    recipe_lines = recipe_section.split("---")[0].strip().splitlines()
    records = [json.loads(line) for line in recipe_lines]
    assert records[0]["action"] == "pi0_nav_pick"
    assert "action" not in records[1]
    assert records[1]["_standalone_only_step"]["action"] == "press"


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
            "bridge_memory_excerpt": "(no published task memory yet)",
        },
    )
    assert "navigate_to" in prompt
    assert "unavailable" in prompt
    assert "pi0_nav_pick" in prompt
