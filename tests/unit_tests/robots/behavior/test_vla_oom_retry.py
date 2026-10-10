"""Contract tests for the BEHAVIOR VLA OOM-recovery helper."""

from __future__ import annotations

import numpy as np
import torch

from robots.behavior.vla_server import _predict_with_oom_retry


class _FakeOomModel:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def predict_action_batch(self, obs, mode="eval", compute_values=False):
        self.calls += 1
        if self.calls <= self.failures:
            raise torch.OutOfMemoryError("simulated CUDA OOM")
        return np.zeros((1, 32, 23), dtype=np.float32), None


def test_predict_retries_once_after_oom() -> None:
    model = _FakeOomModel(failures=1)
    actions, _ = _predict_with_oom_retry(model, {})
    assert model.calls == 2
    assert actions.shape == (1, 32, 23)


def test_predict_propagates_repeated_oom() -> None:
    model = _FakeOomModel(failures=2)
    try:
        _predict_with_oom_retry(model, {})
    except torch.OutOfMemoryError:
        assert model.calls == 2
    else:  # pragma: no cover - must not reach here
        raise AssertionError("second OOM must propagate")
