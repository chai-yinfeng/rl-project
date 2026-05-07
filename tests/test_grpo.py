import sys
import types

import pytest

from reasoning_post_training.methods.grpo import build_grpo_config, compute_group_relative_advantages


def test_compute_group_relative_advantages_normalizes_within_group():
    advantages = compute_group_relative_advantages([1.0, 2.0, 3.0, 4.0], group_size=2)

    assert advantages == pytest.approx([-1.0, 1.0, -1.0, 1.0])


def test_compute_group_relative_advantages_returns_zero_for_tied_group():
    advantages = compute_group_relative_advantages([1.0, 1.0], group_size=2)

    assert advantages == [0.0, 0.0]


def test_compute_group_relative_advantages_rejects_incomplete_group():
    with pytest.raises(ValueError, match="divisible"):
        compute_group_relative_advantages([1.0, 2.0, 3.0], group_size=2)


def test_build_grpo_config_defaults_generation_batch_size(monkeypatch):
    class FakeGRPOConfig:
        def __init__(
            self,
            num_generations=2,
            per_device_train_batch_size=1,
            generation_batch_size=None,
        ):
            self.num_generations = num_generations
            self.per_device_train_batch_size = per_device_train_batch_size
            self.generation_batch_size = generation_batch_size

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(GRPOConfig=FakeGRPOConfig))

    grpo_config = build_grpo_config({"num_generations": 2, "per_device_train_batch_size": 1})

    assert grpo_config.generation_batch_size == 2


def test_build_grpo_config_rejects_invalid_generation_batch_size(monkeypatch):
    class FakeGRPOConfig:
        def __init__(
            self,
            num_generations=2,
            per_device_train_batch_size=1,
            generation_batch_size=None,
        ):
            pass

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(GRPOConfig=FakeGRPOConfig))

    with pytest.raises(ValueError, match="generation_batch_size"):
        build_grpo_config({"num_generations": 2, "generation_batch_size": 3})
