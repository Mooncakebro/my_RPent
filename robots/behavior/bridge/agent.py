"""RPent planner/session factory used by the official evaluator bridge."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from robots.behavior import prompt_bundle
from robots.behavior.prompts.bridge_system import system_prompt as bridge_system_prompt
from robots.behavior.task_specs import get_task_spec
from robots.behavior.toolkit import BehaviorToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.base import build_planner
from rpent.prompt.utils import format_prompt
from rpent.robots.runtime import stop_owned_daemons
from rpent.utils.rpc import wait_for_ready


class BridgedAgent:
    """Own one planner, toolkit, VLA sidecar, and episode lifecycle."""

    def __init__(
        self,
        env: Any,
        *,
        planner: str = "api",
        model: str | None = None,
        task_name: str = "turning_on_radio",
        public_seed: int = 0,
        output_dir: str | Path = "bridge-output",
        memory: str | Path | None = None,
        memory_dir: str | Path | None = None,
        memory_profile: str = "hf",
        checkpoint: str | Path | None = None,
        policy_checkpoint: str | Path | None = None,
        behavior_python: str | Path | None = None,
        behavior_repo: str | Path | None = None,
        vla_endpoint: str | None = None,
        cuda_devices: str | None = None,
        cuda_device: str | None = None,
        max_turns: int | None = None,
        planner_timeout_s: float | None = None,
        max_episode_steps: int = 43_200,
        no_images: bool = False,
        **_: Any,
    ) -> None:
        self.env = env
        self.task_name = str(task_name)
        self.public_seed = int(public_seed)
        self.output_dir = Path(output_dir or "bridge-output").expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir = Path(memory_dir or memory or (self.output_dir / "memory")).expanduser().resolve()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_profile = memory_profile
        self.max_turns = int(max_turns or 64)
        self.planner_timeout_s = planner_timeout_s
        self.no_images = no_images
        self._owned_daemons: dict[str, Any] = {}
        self._dashboard_events = NullDashboardEventSink()
        self._closed = False
        self.toolkit: BehaviorToolkit | None = None

        spec = get_task_spec(self.task_name)
        if self.public_seed not in spec.public_seed_to_instance:
            raise ValueError(f"public seed {self.public_seed} is invalid for {self.task_name!r}")
        self.spec = spec
        self.task_language = spec.task_language

        # Reuse the existing behavior-v env/VLA launcher. Only the VLA is
        # owned here: OmniGibson remains in the official evaluator process.
        from robots.behavior import runtime

        selected_checkpoint = policy_checkpoint or checkpoint
        if vla_endpoint is None and not selected_checkpoint:
            raise ValueError("bridge agent requires --policy-checkpoint or --vla-endpoint")
        if vla_endpoint is None and not behavior_python:
            raise ValueError("bridge agent requires --behavior-python when spawning VLA")
        args = argparse.Namespace(
            vla_endpoint=vla_endpoint,
            policy_checkpoint=str(selected_checkpoint or ""),
            behavior_python=str(behavior_python or ""),
            behavior_model_cuda_device=cuda_devices or cuda_device,
            cuda_device=cuda_devices or cuda_device,
        )
        daemon, rpc = runtime._spawn_vla_server(args, self.output_dir)
        if daemon is not None:
            self._owned_daemons["vla"] = daemon
            wait_for_ready(rpc, daemon=daemon, timeout_s=900.0)
        self.vla = runtime._connect_vla(args, rpc, self.output_dir)["model"]

        self.memory = MemoryManager(
            self.memory_dir,
            memory_access="read_only",
        )
        config = SimpleNamespace(
            output_dir=self.output_dir,
            prompt_vars={
                "task_name": spec.task_name,
                "task": spec.task_index,
                "task_language": spec.task_language,
                "instruction": spec.task_language,
                "task_instruction": spec.task_language,
                "public_seed": self.public_seed,
                "activity_definition_id": spec.activity_definition_id,
                "activity_instance_id": spec.public_seed_to_instance[self.public_seed],
                "scene_model": spec.scene_model,
                "mode": "bridge",
                "behavior_phase": "eval",
                "max_episode_steps": int(max_episode_steps),
                "wall_clock_seconds": int(planner_timeout_s or 7200),
                "memory_dir": str(self.memory_dir),
                "memory_profile": memory_profile,
                "memory_inbox": str(self.memory_dir / "_internal" / "inbox" / "bridge"),
                "recipe_tag": f"bridge_{spec.task_name}_s{self.public_seed}",
                "output_dir": str(self.output_dir),
                "public_capabilities": [
                    "observe",
                    "pixel_to_world",
                    "pi0_nav_pick",
                    "open",
                    "close",
                    "finish",
                ],
            },
        )
        runtime_kwargs = {
            "env": env,
            "model": self.vla,
            "task_name": spec.task_name,
            "public_seed": self.public_seed,
            "max_episode_steps": int(max_episode_steps),
            "output_dir": self.output_dir,
            "memory_dir": self.memory_dir,
            "behavior_phase": "eval",
            "initial_observation": None,
            "initial_info": None,
        }
        self.toolkit = BehaviorToolkit(
            runtime_kwargs=runtime_kwargs,
            dashboard_events=self._dashboard_events,
            memory=self.memory,
            config=config,
            state_output_dir=self.output_dir,
        )
        # The toolkit's standard registration is intentionally broad for
        # normal RPent runs. Remove simulator-state motion tools in bridged
        # mode so the planner cannot spend turns invoking known failures.
        for name in ("navigate_to", "move_to", "rotate_wrist", "press"):
            self.toolkit._tools.pop(name, None)

        prompt_vars = config.prompt_vars
        self.system_prompt = format_prompt(bridge_system_prompt(), variables=prompt_vars)
        self.user_prompt = format_prompt(
            prompt_bundle.user_prompt(variables=prompt_vars),
            variables=prompt_vars,
        )
        self.planner = build_planner(
            planner,
            output_dir=self.output_dir,
            recipe_tag=prompt_vars["recipe_tag"],
            robot_name="behavior",
            model=model,
            memory_dir=self.memory_dir,
            planner_timeout_s=planner_timeout_s,
            dashboard_events=self._dashboard_events,
            no_images=no_images,
        )

    def run(self) -> Any:
        """Run the standard planner loop until finish, timeout, or failure."""

        try:
            return self.planner.solve(
                system_prompt=self.system_prompt,
                user_message=self.user_prompt,
                toolkit=self.toolkit,
                max_turns=self.max_turns,
            )
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.toolkit is not None:
                self.toolkit.close()
        finally:
            stop_owned_daemons(self._owned_daemons, self._dashboard_events)
            self._owned_daemons.clear()

    def __del__(self) -> None:  # pragma: no cover - defensive partial-init cleanup
        try:
            self.close()
        except Exception:
            pass


def factory(env: Any, **kwargs: Any) -> BridgedAgent:
    """Construct the built-in bridge agent for ``--agent-factory`` use."""

    return BridgedAgent(env, **kwargs)


__all__ = ["BridgedAgent", "factory"]
