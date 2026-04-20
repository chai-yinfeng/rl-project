import pytest

from grpo_reasoning.algorithms.grpo import compute_group_relative_advantages


def test_compute_group_relative_advantages_normalizes_within_group():
    advantages = compute_group_relative_advantages([1.0, 2.0, 3.0, 4.0], group_size=2)

    assert advantages == pytest.approx([-1.0, 1.0, -1.0, 1.0])


def test_compute_group_relative_advantages_returns_zero_for_tied_group():
    advantages = compute_group_relative_advantages([1.0, 1.0], group_size=2)

    assert advantages == [0.0, 0.0]


def test_compute_group_relative_advantages_rejects_incomplete_group():
    with pytest.raises(ValueError, match="divisible"):
        compute_group_relative_advantages([1.0, 2.0, 3.0], group_size=2)

