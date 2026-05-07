import sys
import types

from reasoning_post_training.methods.dpo import (
    build_gold_chosen_pair,
    choose_preference_pair,
    load_dpo_jsonl,
    normalize_dpo_completion,
    score_gsm8k_completion,
)


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


def test_build_gold_chosen_pair_uses_gold_solution_as_chosen():
    pair = build_gold_chosen_pair("Reasoning. Final answer: 12", "We compute it. #### 42", "42")

    assert pair["chosen_score"] > pair["rejected_score"]
    assert pair["chosen_predicted_answer"] == "42"


def test_normalize_dpo_completion_adds_response_boundary():
    assert normalize_dpo_completion("Reasoning") == " Reasoning"
    assert normalize_dpo_completion(" Reasoning") == " Reasoning"


def test_load_dpo_jsonl_normalizes_completion_boundaries(tmp_path, monkeypatch):
    class FakeDataset:
        @classmethod
        def from_list(cls, records):
            return records

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=FakeDataset))
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"prompt": "Solution:", "chosen": "Correct", "rejected": "Wrong"}\n',
        encoding="utf-8",
    )

    record = load_dpo_jsonl(path)[0]

    assert record["chosen"] == " Correct"
    assert record["rejected"] == " Wrong"
