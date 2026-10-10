# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BEHAVIOR preset and inference adapter for the shared Pi0.5 facade."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import rpent.robots.base  # noqa: F401

from robots.behavior.schemas import validate_action_chunk
from rpent.robots.components.pi05_vla_server import (
    PI05_EMBODIMENTS,
    PI05_ROBOT_PLATFORMS,
    Pi05VLAFacade,
)
from rpent.utils.serialization import to_numpy_tree

PI05_EMBODIMENTS["behavior"] = {
    "seed": 0,
    "num_action_chunks": 32,
    "action_dim": 32,
    "use_proprio": True,
    "num_steps": 4,
    "add_value_head": False,
    "openpi_data": {
        "norm_stats_path": "assets/behavior-1k/2025-challenge-demos/norm_stats.json",
        "extra_delta_transform": False,
        "extract_state_from_proprio": True,
        "use_all_wrist_images": True,
        "use_quantile_norm": True,
    },
    "openpi": {
        "config_name": "pi05_behavior",
        "num_images_in_input": 3,
        "action_dim": 32,
        "action_horizon": 32,
        "action_chunk": 32,
        "action_env_dim": 23,
        "num_steps": 4,
        "add_value_head": False,
        "noise_level": 0.0,
        "noise_method": "flow_sde",
        "joint_logprob": False,
    },
}
PI05_ROBOT_PLATFORMS["behavior"] = "BEHAVIOR"


def _predict_with_oom_retry(model: Any, obs: dict) -> Any:
    """One action-chunk prediction with a single OOM recovery attempt.

    The first inference after model load is the peak allocation of an episode
    (32-chunk generation right after ``.cuda()`` leaves the allocator
    fragmented). On a shared GPU that is often the difference between success
    and ``torch.OutOfMemoryError``. Free the cached-but-unused blocks and retry
    once; a second failure is a genuine capacity problem and propagates.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        with torch.no_grad():
            return model.predict_action_batch(obs, mode="eval", compute_values=False)
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.no_grad():
            return model.predict_action_batch(obs, mode="eval", compute_values=False)


class BehaviorPi05VLAFacade(Pi05VLAFacade):
    def __init__(
        self,
        *,
        model_path: str,
        embodiment: str,
        model_backend: str = "openpi_pytorch",
        norm_stats_path: str | None = None,
        repo_id: str | None = None,
    ):
        if embodiment != "behavior":
            raise ValueError("BEHAVIOR facade requires the behavior embodiment")
        # The preset keeps RLinf's BEHAVIOR observation transforms. The common
        # norm_stats_path override replaces that entire config, so resolve here.
        config = PI05_EMBODIMENTS["behavior"]
        config["openpi_data"]["norm_stats_path"] = str(
            Path(model_path) / "assets/behavior-1k/2025-challenge-demos/norm_stats.json"
        )
        super().__init__(
            model_path=model_path,
            embodiment=embodiment,
            model_backend=model_backend,
            norm_stats_path=norm_stats_path,
            repo_id=repo_id,
        )
        # Drop cached-but-unused allocator blocks left over from loading the
        # checkpoint onto CUDA; the first inference is the peak allocation.
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, obs: dict, options: dict | None = None) -> np.ndarray:
        import torch

        if options is not None and not isinstance(options, dict):
            raise TypeError("VLA options must be a mapping")
        if (
            set(options or {}) - {"mode"}
            or (options or {}).get("mode", "eval") != "eval"
        ):
            raise ValueError("BEHAVIOR VLA only supports eval mode")
        # The checkpoint has no value head; the pinned RLinf loader defaults
        # compute_values to True. RpcFacade already serializes this request.
        actions, _ = _predict_with_oom_retry(self._model, obs)
        result = np.asarray(to_numpy_tree(actions), dtype=np.float32)
        if result.ndim != 3 or result.shape[0] != 1:
            raise ValueError(f"BEHAVIOR actions must be [1,T,23], got {result.shape}")
        validate_action_chunk(result[0])
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=None)
    args = parser.parse_args()
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    BehaviorPi05VLAFacade(model_path=args.model_path, embodiment="behavior").serve(
        transport="http",
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
