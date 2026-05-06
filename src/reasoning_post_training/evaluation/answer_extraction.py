"""Answer extraction helpers for GSM8K-style math responses."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


_GSM8K_FINAL_RE = re.compile(r"####\s*([-+]?(?:\d[\d,]*)(?:\.\d+)?)")
_LABELED_FINAL_RE = re.compile(
    r"(?:final answer|answer|therefore|so the answer is)\s*[:=]?\s*"
    r"([-+]?(?:\d[\d,]*)(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?")


def normalize_numeric_answer(value: str | int | float | Decimal | None) -> str | None:
    """Normalize numeric strings so equivalent answers compare equal."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.replace(",", "")
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None

    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))

    return format(number.normalize(), "f")


def extract_gsm8k_gold_answer(answer_text: str) -> str | None:
    """Extract the canonical final answer from a GSM8K answer field."""
    match = _GSM8K_FINAL_RE.search(answer_text)
    if match:
        return normalize_numeric_answer(match.group(1))
    return extract_predicted_answer(answer_text)


def extract_predicted_answer(completion: str) -> str | None:
    """Extract a final numeric answer from a model completion."""
    labeled_matches = list(_LABELED_FINAL_RE.finditer(completion))
    if labeled_matches:
        return normalize_numeric_answer(labeled_matches[-1].group(1))

    numbers = _NUMBER_RE.findall(completion)
    if not numbers:
        return None
    return normalize_numeric_answer(numbers[-1])


def is_exact_match(completion: str, gold_answer: str | int | float | Decimal | None) -> bool:
    """Return whether a completion's extracted answer matches the gold answer."""
    prediction = extract_predicted_answer(completion)
    gold = normalize_numeric_answer(gold_answer)
    return prediction is not None and gold is not None and prediction == gold

