from grpo_reasoning.evaluation.answer_extraction import (
    extract_gsm8k_gold_answer,
    extract_predicted_answer,
    is_exact_match,
    normalize_numeric_answer,
)


def test_normalize_numeric_answer_removes_commas_and_trailing_zeroes():
    assert normalize_numeric_answer("1,234.0") == "1234"
    assert normalize_numeric_answer("03.50") == "3.5"


def test_extract_gsm8k_gold_answer_prefers_hash_marker():
    assert extract_gsm8k_gold_answer("Some work here. #### 42") == "42"


def test_extract_predicted_answer_prefers_labeled_final_answer():
    completion = "We compute several values: 10, 15. Final answer: 12"
    assert extract_predicted_answer(completion) == "12"


def test_extract_predicted_answer_falls_back_to_last_number():
    assert extract_predicted_answer("The result is therefore 7.") == "7"


def test_is_exact_match_handles_equivalent_numbers():
    assert is_exact_match("Final answer: 1,200.0", "1200")

