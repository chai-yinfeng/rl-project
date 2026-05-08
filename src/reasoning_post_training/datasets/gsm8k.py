"""GSM8K dataset loading and prompt formatting."""

from __future__ import annotations

from dataclasses import dataclass

from reasoning_post_training.evaluation.answer_extraction import extract_gsm8k_gold_answer


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem step by step, then end with "
    "'Final answer: <number>'."
)


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    answer: str
    gold_answer: str | None
    prompt: str


def format_gsm8k_prompt(question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Build a plain text prompt that works across chat and causal LM baselines."""
    return f"{system_prompt}\n\nProblem:\n{question.strip()}\n\nSolution:"


def split_gsm8k_prompt(prompt: str) -> tuple[str, str]:
    """Split a project GSM8K prompt into system and user chat messages."""
    marker = "\n\nProblem:\n"
    if marker not in prompt:
        return DEFAULT_SYSTEM_PROMPT, prompt.strip()
    system_prompt, problem_part = prompt.split(marker, 1)
    if problem_part.endswith("\n\nSolution:"):
        problem_part = problem_part[: -len("\n\nSolution:")]
    return system_prompt.strip(), f"Problem:\n{problem_part.strip()}"


def format_gsm8k_chat_prompt(tokenizer, prompt: str) -> str:
    """Apply a tokenizer chat template when available, matching Qwen Instruct usage."""
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        return prompt

    system_prompt, user_prompt = split_gsm8k_prompt(prompt)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def convert_gsm8k_record(record: dict[str, str]) -> GSM8KExample:
    """Convert a raw GSM8K record into the project schema."""
    question = record["question"]
    answer = record["answer"]
    return GSM8KExample(
        question=question,
        answer=answer,
        gold_answer=extract_gsm8k_gold_answer(answer),
        prompt=format_gsm8k_prompt(question),
    )


def load_gsm8k_split(split: str = "test", subset: str = "main"):
    """Load a GSM8K split through Hugging Face datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before loading GSM8K.") from exc

    dataset = load_dataset("gsm8k", subset, split=split)
    return dataset.map(lambda record: convert_gsm8k_record(record).__dict__)
