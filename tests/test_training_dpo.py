from grpo_reasoning.training.dpo import choose_preference_pair, score_gsm8k_completion


def test_score_gsm8k_completion_combines_correctness_and_format():
    assert score_gsm8k_completion("Reasoning. Final answer: 42", "42") == 1.1
    assert score_gsm8k_completion("No numeric answer here", "42") == -0.1


def test_choose_preference_pair_uses_best_and_worst_scores():
    pair = choose_preference_pair(
        [
            "Reasoning. Final answer: 12",
            "Reasoning. Final answer: 42",
            "No final numeric answer",
        ],
        "42",
    )

    assert pair is not None
    assert pair["chosen"] == "Reasoning. Final answer: 42"
    assert pair["rejected"] == "No final numeric answer"
    assert pair["chosen_score"] > pair["rejected_score"]


def test_choose_preference_pair_can_skip_ties():
    assert choose_preference_pair(["Final answer: 1", "Final answer: 2"], "42") is None
