"""Answer extraction helpers for GSM8K-style math responses."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


_GSM8K_FINAL_RE = re.compile(r"####\s*([-+]?(?:\d[\d,]*)(?:\.\d+)?)")
_FINAL_ANSWER_RE = re.compile(
    r"final answer\s*[:=]?\s*([-+]?(?:\d[\d,]*)(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
_LABELED_FINAL_RE = re.compile(
    r"(?:the answer is|answer|therefore|so the answer is)\s*[:=]?\s*"
    r"([-+]?(?:\d[\d,]*)(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?")
_GENERATION_LEAK_MARKERS = (
    "\nHuman:",
    "\nAssistant:",
    "\nUser:",
    "\nSystem:",
    "<|im_start|>",
    "<|im_end|>",
)


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
        try:
            return str(number.quantize(Decimal(1)))
        except InvalidOperation:
            return format(number, "f")

    try:
        return format(number.normalize(), "f")
    except InvalidOperation:
        return format(number, "f")


def extract_gsm8k_gold_answer(answer_text: str) -> str | None:
    """Extract the canonical final answer from a GSM8K answer field."""
    match = _GSM8K_FINAL_RE.search(answer_text)
    if match:
        return normalize_numeric_answer(match.group(1))
    return extract_predicted_answer(answer_text)


def extract_predicted_answer(completion: str) -> str | None:
    """Extract a final numeric answer from a model completion."""
    completion = truncate_completion(completion)

    final_match = _FINAL_ANSWER_RE.search(completion)
    if final_match:
        return normalize_numeric_answer(final_match.group(1))

    labeled_matches = list(_LABELED_FINAL_RE.finditer(completion))
    if labeled_matches:
        return normalize_numeric_answer(labeled_matches[0].group(1))

    numbers = _NUMBER_RE.findall(completion)
    if not numbers:
        return None
    return normalize_numeric_answer(numbers[-1])


def truncate_completion(completion: str) -> str:
    """Remove obvious prompt/chat leakage after the model's answer."""
    text = str(completion).strip()
    if not text:
        return text

    cut = len(text)
    final_match = _FINAL_ANSWER_RE.search(text)
    if final_match:
        cut = min(cut, final_match.end())
    for marker in _GENERATION_LEAK_MARKERS:
        marker_index = text.find(marker)
        if marker_index >= 0:
            cut = min(cut, marker_index)
    return text[:cut].strip()


def is_exact_match(completion: str, gold_answer: str | int | float | Decimal | None) -> bool:
    """Return whether a completion's extracted answer matches the gold answer."""
    prediction = extract_predicted_answer(completion)
    gold = normalize_numeric_answer(gold_answer)
    return prediction is not None and gold is not None and prediction == gold
