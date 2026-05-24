from __future__ import annotations

from fractions import Fraction
import re


ANSWER_PATTERNS = [
    re.compile(r"Answer\s*:\s*([-+]?\d+(?:/\d+)?(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\\boxed\{([-+]?\d+(?:/\d+)?(?:\.\d+)?)\}"),
]


def extract_answer(text: str) -> str | None:
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()

    numbers = re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", text)
    if not numbers:
        return None
    return numbers[-1].strip()


def normalize_number(value: str | None) -> Fraction | None:
    if value is None:
        return None
    value = value.strip().replace(",", "")
    try:
        return Fraction(value)
    except ValueError:
        try:
            return Fraction(float(value)).limit_denominator(1000000)
        except ValueError:
            return None


def is_correct(predicted: str | None, gold: str) -> bool:
    pred = normalize_number(predicted)
    target = normalize_number(gold)
    return pred is not None and target is not None and pred == target


def correctness_reward(response: str, gold: str) -> float:
    return 1.0 if is_correct(extract_answer(response), gold) else 0.0

