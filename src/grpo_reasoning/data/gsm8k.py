"""GSM8K dataset loading and prompt formatting."""

from __future__ import annotations

from dataclasses import dataclass

from grpo_reasoning.evaluation.answer_extraction import extract_gsm8k_gold_answer


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

