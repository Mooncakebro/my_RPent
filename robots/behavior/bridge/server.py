"""Command-line server for running an RPent Behavior agent through the bridge."""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .protocol import WebsocketBridgeServer, sanitize_action
from .shim_env import BridgeRendezvous, ShimBehaviorEnv, adapt_observation

LOGGER = logging.getLogger(__name__)


def _load_factory(spec: str | None) -> Callable[..., Any] | None:
    if not spec:
        return None
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--agent-factory must use MODULE:ATTRIBUTE syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"agent factory is not callable: {spec}")
    return factory


class AgentSessionPolicy:
    """Turn a planner callable into a websocket ``obs -> action`` policy.

    The planner runs in one worker thread and interacts with ``ShimBehaviorEnv``
    exactly as it does with RPent's normal env client.  The websocket callback
    only performs the capacity-one rendezvous and therefore preserves evaluator
    lockstep semantics.
    """

    def __init__(
        self,
        *,
        agent_factory: Callable[..., Any] | None = None,
        task_language: str = "",
        agent_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.agent_factory = agent_factory
        self.task_language = task_language
        self.agent_kwargs = dict(agent_kwargs or {})
        self._session_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._error: BaseException | None = None
        self._agent: Any | None = None
        self.env: ShimBehaviorEnv | None = None
        self.rendezvous: BridgeRendezvous | None = None
        self._latest_action = np.zeros(23, dtype=np.float32)
        self._latest_observation: dict[str, Any] | None = None

    def reset(self) -> None:
        with self._session_lock:
            old_agent = self._agent
            if old_agent is not None:
                closer = getattr(old_agent, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        LOGGER.exception("failed to close previous bridge agent")
            if self.rendezvous is not None:
                self.rendezvous.close()
            self.rendezvous = BridgeRendezvous()
            self.rendezvous.reset()
            self.env = ShimBehaviorEnv(self.rendezvous, task_language=self.task_language)
            self._done.clear()
            self._error = None
            self._latest_action = np.zeros(23, dtype=np.float32)
            self._thread = threading.Thread(target=self._run_agent, name="behavior-agent", daemon=True)
            self._thread.start()

    def _run_agent(self) -> None:
        try:
            if self.agent_factory is None:
                self._done.set()
                return
            assert self.env is not None
            try:
                signature = inspect.signature(self.agent_factory)
            except (TypeError, ValueError):
                signature = None
            if signature is None:
                # Some extension callables do not expose an inspectable
                # signature; retain the simple factory(env) contract.
                agent = self.agent_factory(self.env)
            else:
                accepts_kwargs = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
                kwargs = (
                    self.agent_kwargs
                    if accepts_kwargs
                    else {
                        name: value
                        for name, value in self.agent_kwargs.items()
                        if name in signature.parameters
                    }
                )
                agent = (
                    self.agent_factory(self.env, **kwargs)
                    if signature.parameters
                    else self.agent_factory(**kwargs)
                )
            self._agent = agent
            if callable(agent):
                agent()
            elif hasattr(agent, "run") and callable(agent.run):
                agent.run()
        except BaseException as exc:  # make the websocket fail safe, not stuck
            self._error = exc
            LOGGER.exception("bridge agent thread failed")
        finally:
            self._agent = None
            self._done.set()

    def _hold_for(self, observation: Mapping[str, Any]) -> np.ndarray:
        try:
            adapted = adapt_observation(observation, task_language=self.task_language)
            self._latest_observation = adapted
            if self.env is not None:
                self.env.last_observation = adapted
                self.env.last_info = {"_bridge": True}
                return self.env._hold_action_from_current_proprio()
        except Exception:
            pass
        return sanitize_action(self._latest_action)

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        rendezvous = self.rendezvous
        if rendezvous is None or self._thread is None:
            self.reset()
            rendezvous = self.rendezvous
        assert rendezvous is not None
        if self._done.is_set():
            return self._hold_for(observation)
        # The worker's env.reset()/step() consumes exactly one observation.
        rendezvous.put_observation(observation)
        while True:
            try:
                action = rendezvous.actions.get(timeout=0.25)
                break
            except Exception:
                if self._done.is_set():
                    return self._hold_for(observation)
        self._latest_action = sanitize_action(action, current=self._latest_action)
        return self._latest_action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-language", default="")
    parser.add_argument("--agent-factory", help="planner factory as MODULE:ATTRIBUTE")
    # Keep the RPent command-line surface available to launch scripts.  The
    # bridge itself does not inspect privileged simulator state.
    parser.add_argument("--planner", default="api")
    parser.add_argument("--model", default=None)
    parser.add_argument("--memory", default=None)
    parser.add_argument("--memory-profile", default="hf")
    parser.add_argument("--checkpoint", "--policy-checkpoint", dest="checkpoint", default=None)
    parser.add_argument("--cuda-devices", "--cuda-device", dest="cuda_devices", default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--planner-timeout-s", type=float, default=None)
    parser.add_argument("--task-name", default="turning_on_radio")
    parser.add_argument("--public-seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--behavior-python", default=None)
    parser.add_argument("--behavior-repo", default=None)
    parser.add_argument("--vla-endpoint", default=None)
    parser.add_argument("--max-episode-steps", type=int, default=43200)
    parser.add_argument("--metadata-json", default=None)
    return parser


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "runtime": "rpent_behavior_bridge",
        "protocol": "behavior_websocket_v1",
        "action_dim": 23,
        "task_language": args.task_language,
    }
    if args.metadata_json:
        import json

        with open(args.metadata_json, encoding="utf-8") as source:
            extra = json.load(source)
        if not isinstance(extra, dict):
            raise ValueError("--metadata-json must contain a JSON object")
        metadata.update(extra)
    return metadata


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    if not args.task_language and args.task_name:
        from robots.behavior.task_specs import get_task_spec

        args.task_language = get_task_spec(args.task_name).task_language
    from .agent import factory as built_in_factory

    policy = AgentSessionPolicy(
        agent_factory=_load_factory(args.agent_factory) or built_in_factory,
        task_language=args.task_language,
        agent_kwargs={
            "planner": args.planner,
            "model": args.model,
            "memory": args.memory,
            "memory_profile": args.memory_profile,
            "checkpoint": args.checkpoint,
            "cuda_devices": args.cuda_devices,
            "max_turns": args.max_turns,
            "planner_timeout_s": args.planner_timeout_s,
            "task_name": args.task_name,
            "public_seed": args.public_seed,
            "output_dir": args.output_dir,
            "behavior_python": args.behavior_python,
            "behavior_repo": args.behavior_repo,
            "vla_endpoint": args.vla_endpoint,
            "max_episode_steps": args.max_episode_steps,
        },
    )
    WebsocketBridgeServer(
        policy,
        host=args.host,
        port=args.port,
        metadata=_metadata(args),
    ).serve_forever()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["AgentSessionPolicy", "build_parser", "main"]
