from reasoning_post_training.evaluation.metrics import evaluate_completions
from reasoning_post_training.methods.baseline import prompt_hash


def test_evaluate_completions_counts_accuracy_and_invalid_rate():
    result = evaluate_completions(
        ["Final answer: 4", "Final answer: 5", "No numeric answer"],
        ["4", "6", "7"],
    )

    assert result.total == 3
    assert result.correct == 1
    assert result.invalid == 1
    assert result.accuracy == 1 / 3
    assert result.invalid_rate == 1 / 3


def test_prompt_hash_changes_with_prompt_text():
    assert prompt_hash("old prompt") != prompt_hash("new prompt")
