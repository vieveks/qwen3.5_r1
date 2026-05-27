from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from iso_rlvr.rewards.answer import is_correct
from iso_rlvr.rewards.packed_answer import PackedAnswerParse, parse_packed_answers


@dataclass(frozen=True)
class PackedRewardConfig:
    format_reward: float = 0.05
    missing_format_penalty: float = -0.10
    family_mean_weight: float = 0.25
    all_family_correct_weight: float = 0.25
    extra_answer_penalty: float = 0.10
    length_penalty_weight: float = 0.05
    token_cap: int = 256


@dataclass(frozen=True)
class PackedRewardBreakdown:
    reward: float
    parse: PackedAnswerParse
    correctness: list[bool]
    family_mean: float
    all_family_correct: bool
    format_component: float
    family_component: float
    extra_answer_penalty: float
    length_penalty: float


def score_packed_completion(
    completion: str,
    gold_answers: Sequence[str],
    response_tokens: int | None = None,
    config: PackedRewardConfig | None = None,
) -> PackedRewardBreakdown:
    if not gold_answers:
        raise ValueError("gold_answers must not be empty.")

    cfg = config or PackedRewardConfig()
    parsed = parse_packed_answers(completion, expected_count=len(gold_answers))
    correctness = [
        is_correct(predicted, gold) for predicted, gold in zip(parsed.answers, gold_answers)
    ]
    correct_count = sum(correctness)
    family_mean = correct_count / len(gold_answers)
    all_family_correct = family_mean == 1.0

    format_values = [
        cfg.format_reward if answer is not None else cfg.missing_format_penalty
        for answer in parsed.answers
    ]
    family_values = [
        (
            cfg.family_mean_weight * family_mean
            + cfg.all_family_correct_weight * float(all_family_correct)
        )
        if correct
        else 0.0
        for correct in correctness
    ]
    per_variant = [
        float(correct) + family_value + format_value
        for correct, family_value, format_value in zip(correctness, family_values, format_values)
    ]

    extra_penalty = cfg.extra_answer_penalty * len(parsed.extra_answers) / len(gold_answers)
    length_penalty = _wrong_length_penalty(
        correct_count=correct_count,
        total_count=len(gold_answers),
        response_tokens=response_tokens,
        config=cfg,
    )
    reward = sum(per_variant) / len(per_variant) - extra_penalty - length_penalty

    return PackedRewardBreakdown(
        reward=reward,
        parse=parsed,
        correctness=correctness,
        family_mean=family_mean,
        all_family_correct=all_family_correct,
        format_component=sum(format_values) / len(format_values),
        family_component=sum(family_values) / len(family_values),
        extra_answer_penalty=extra_penalty,
        length_penalty=length_penalty,
    )


def packed_reward_values(
    completions: Sequence[str],
    gold_answers: Sequence[Sequence[str]],
    response_tokens: Sequence[int | None] | None = None,
    config: PackedRewardConfig | None = None,
) -> list[float]:
    if len(completions) != len(gold_answers):
        raise ValueError("completions and gold_answers must have the same length.")
    if response_tokens is not None and len(response_tokens) != len(completions):
        raise ValueError("response_tokens must match completions length.")

    token_values = response_tokens or [None] * len(completions)
    return [
        score_packed_completion(
            completion,
            answers,
            response_tokens=tokens,
            config=config,
        ).reward
        for completion, answers, tokens in zip(completions, gold_answers, token_values)
    ]


def _wrong_length_penalty(
    correct_count: int,
    total_count: int,
    response_tokens: int | None,
    config: PackedRewardConfig,
) -> float:
    wrong_count = total_count - correct_count
    if wrong_count == 0 or response_tokens is None:
        return 0.0
    capped_tokens = min(max(response_tokens, 0), config.token_cap)
    wrong_fraction = wrong_count / total_count
    return config.length_penalty_weight * wrong_fraction * capped_tokens / config.token_cap
