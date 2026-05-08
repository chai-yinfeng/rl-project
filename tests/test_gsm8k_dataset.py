from reasoning_post_training.datasets.gsm8k import DEFAULT_SYSTEM_PROMPT, format_gsm8k_prompt


def test_default_system_prompt_requires_final_answer_last_line():
    assert "Final answer: <number>" in DEFAULT_SYSTEM_PROMPT
    assert "Do not write anything after the final answer." in DEFAULT_SYSTEM_PROMPT


def test_format_gsm8k_prompt_includes_strong_final_answer_instruction():
    prompt = format_gsm8k_prompt("What is 2 + 2?")

    assert "Final answer: <number>" in prompt
    assert prompt.endswith("Solution:")
