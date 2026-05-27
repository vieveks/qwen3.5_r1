from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from iso_rlvr.rewards.answer import normalize_number
from iso_rlvr.rewards.packed_answer import PackedAnswerParse, parse_packed_answers


@dataclass(frozen=True)
class PackedOutputDiagnostics:
    repeated_answer: bool
    copied_answer_indices: list[int]
    only_first_answer: bool
    missing_indices: list[int]
    extra_answer_count: int
    answer_count_mismatch: bool
    same_wrong_additive_offset: bool
    same_wrong_multiplicative_offset: bool

    @property
    def suspicious(self) -> bool:
        return (
            self.repeated_answer
            or bool(self.copied_answer_indices)
            or self.only_first_answer
            or self.answer_count_mismatch
            or self.same_wrong_additive_offset
            or self.same_wrong_multiplicative_offset
        )


def diagnose_packed_completion(
    completion: str,
    gold_answers: Sequence[str],
) -> PackedOutputDiagnostics:
    parsed = parse_packed_answers(completion, expected_count=len(gold_answers))
    return diagnose_packed_parse(parsed, gold_answers)


def diagnose_packed_parse(
    parsed: PackedAnswerParse,
    gold_answers: Sequence[str],
) -> PackedOutputDiagnostics:
    if len(parsed.answers) != len(gold_answers):
        raise ValueError("parsed answers and gold_answers must have the same length.")

    present_answers = [answer for answer in parsed.answers if answer is not None]
    gold_repeated = _all_normalized_values_same(gold_answers)
    repeated_answer = (
        len(present_answers) > 1 and len(set(present_answers)) == 1 and not gold_repeated
    )
    copied_answer_indices = _copied_answer_indices(parsed.answers, gold_answers)
    only_first_answer = parsed.answers[0] is not None and all(
        answer is None for answer in parsed.answers[1:]
    )
    answer_count_mismatch = bool(parsed.missing_indices or parsed.extra_answers)
    additive, multiplicative = _wrong_offset_flags(parsed.answers, gold_answers)

    return PackedOutputDiagnostics(
        repeated_answer=repeated_answer,
        copied_answer_indices=copied_answer_indices,
        only_first_answer=only_first_answer,
        missing_indices=list(parsed.missing_indices),
        extra_answer_count=len(parsed.extra_answers),
        answer_count_mismatch=answer_count_mismatch,
        same_wrong_additive_offset=additive,
        same_wrong_multiplicative_offset=multiplicative,
    )


def _copied_answer_indices(
    answers: Sequence[str | None],
    gold_answers: Sequence[str],
) -> list[int]:
    seen_answers: set[Fraction | str] = set()
    seen_gold: set[Fraction | str] = set()
    copied = []
    for idx, (answer, gold_raw) in enumerate(zip(answers, gold_answers), start=1):
        gold = _normalized_key(gold_raw)
        if answer is None:
            seen_gold.add(gold)
            continue
        normalized_answer = _normalized_key(answer)
        expected_duplicate = gold in seen_gold
        if normalized_answer in seen_answers and not expected_duplicate:
            copied.append(idx)
        seen_answers.add(normalized_answer)
        seen_gold.add(gold)
    return copied


def _all_normalized_values_same(values: Sequence[str]) -> bool:
    if len(values) < 2:
        return False
    normalized = [_normalized_key(value) for value in values]
    return len(set(normalized)) == 1


def _normalized_key(value: str) -> Fraction | str:
    normalized = normalize_number(value)
    return normalized if normalized is not None else value.strip()


def _wrong_offset_flags(
    answers: Sequence[str | None],
    gold_answers: Sequence[str],
) -> tuple[bool, bool]:
    additive_offsets: list[Fraction] = []
    multiplicative_offsets: list[Fraction] = []

    for predicted_raw, gold_raw in zip(answers, gold_answers):
        predicted = normalize_number(predicted_raw)
        gold = normalize_number(gold_raw)
        if predicted is None or gold is None or predicted == gold:
            continue
        additive_offsets.append(predicted - gold)
        if gold != 0:
            multiplicative_offsets.append(predicted / gold)

    same_additive = (
        len(additive_offsets) > 1
        and len(set(additive_offsets)) == 1
        and additive_offsets[0] != 0
    )
    same_multiplicative = (
        len(multiplicative_offsets) > 1
        and len(set(multiplicative_offsets)) == 1
        and multiplicative_offsets[0] != 1
    )
    return same_additive, same_multiplicative
