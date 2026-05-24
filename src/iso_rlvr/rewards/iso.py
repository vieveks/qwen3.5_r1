from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from iso_rlvr.rewards.answer import extract_answer, is_correct


@dataclass(frozen=True)
class ScoredResponse:
    family_id: str
    variant_id: str
    gold: str
    response: str
    correct: bool
    extracted_answer: str | None
    token_count: int


def score_response(
    family_id: str,
    variant_id: str,
    gold: str,
    response: str,
    token_count: int,
) -> ScoredResponse:
    extracted = extract_answer(response)
    return ScoredResponse(
        family_id=family_id,
        variant_id=variant_id,
        gold=gold,
        response=response,
        correct=is_correct(extracted, gold),
        extracted_answer=extracted,
        token_count=token_count,
    )


def family_consistency(scored: list[ScoredResponse]) -> dict[str, float]:
    by_family: dict[str, list[ScoredResponse]] = defaultdict(list)
    for item in scored:
        by_family[item.family_id].append(item)

    consistency = {}
    for family_id, items in by_family.items():
        consistency[family_id] = 1.0 if items and all(item.correct for item in items) else 0.0
    return consistency


def iso_rewards(scored: list[ScoredResponse], lambda_iso: float) -> dict[str, float]:
    consistency = family_consistency(scored)
    rewards = {}
    for item in scored:
        rewards[item.variant_id] = float(item.correct) + lambda_iso * consistency[item.family_id]
    return rewards


def iso_reward_values(scored: list[ScoredResponse], lambda_iso: float) -> list[float]:
    consistency = family_consistency(scored)
    return [float(item.correct) + lambda_iso * consistency[item.family_id] for item in scored]


def summarize(scored: list[ScoredResponse]) -> dict[str, float]:
    if not scored:
        return {
            "accuracy": 0.0,
            "family_accuracy": 0.0,
            "avg_tokens": 0.0,
            "avg_wrong_tokens": 0.0,
            "format_failure_rate": 0.0,
        }

    consistency = family_consistency(scored)
    wrong = [item for item in scored if not item.correct]
    return {
        "accuracy": sum(item.correct for item in scored) / len(scored),
        "family_accuracy": sum(consistency.values()) / max(len(consistency), 1),
        "avg_tokens": sum(item.token_count for item in scored) / len(scored),
        "avg_wrong_tokens": (
            sum(item.token_count for item in wrong) / len(wrong) if wrong else 0.0
        ),
        "format_failure_rate": sum(item.extracted_answer is None for item in scored) / len(scored),
    }
