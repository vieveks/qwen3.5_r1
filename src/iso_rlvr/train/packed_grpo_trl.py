from __future__ import annotations

from typing import Any, Sequence

from iso_rlvr.rewards.packed_iso import PackedRewardConfig, score_packed_completion


def packed_trl_reward_config(cfg: dict[str, Any] | None = None) -> PackedRewardConfig:
    cfg = cfg or {}
    family_bonus_enabled = bool(cfg.get("family_bonus_enabled", False))
    return PackedRewardConfig(
        format_reward=float(cfg.get("format_reward", 0.05)),
        missing_format_penalty=float(cfg.get("missing_format_penalty", -0.10)),
        family_mean_weight=float(cfg.get("family_mean_weight", 0.25))
        if family_bonus_enabled
        else 0.0,
        all_family_correct_weight=float(cfg.get("all_family_correct_weight", 0.25))
        if family_bonus_enabled
        else 0.0,
        extra_answer_penalty=float(cfg.get("extra_answer_penalty", 0.10)),
        length_penalty_weight=float(cfg.get("length_penalty_weight", 0.05)),
        token_cap=int(cfg.get("token_cap", cfg.get("max_completion_length", 256))),
    )


def make_packed_trl_reward_func(cfg: dict[str, Any] | None = None):
    reward_cfg = packed_trl_reward_config(cfg)

    def reward_func(
        completions: Sequence[Any],
        gold_answers: Sequence[Sequence[str]],
        completion_ids: Sequence[Sequence[int]] | None = None,
        **_kwargs: Any,
    ) -> list[float]:
        return packed_trl_rewards(
            completions=completions,
            gold_answers=gold_answers,
            completion_ids=completion_ids,
            config=reward_cfg,
        )

    return reward_func


def packed_trl_rewards(
    completions: Sequence[Any],
    gold_answers: Sequence[Sequence[str]],
    completion_ids: Sequence[Sequence[int]] | None = None,
    config: PackedRewardConfig | None = None,
) -> list[float]:
    if len(completions) != len(gold_answers):
        raise ValueError("completions and gold_answers must have the same length.")
    if completion_ids is not None and len(completion_ids) != len(completions):
        raise ValueError("completion_ids must match completions length.")

    response_tokens = (
        [len(ids) for ids in completion_ids] if completion_ids is not None else [None] * len(completions)
    )
    return [
        score_packed_completion(
            _completion_to_text(completion),
            answers,
            response_tokens=tokens,
            config=config,
        ).reward
        for completion, answers, tokens in zip(completions, gold_answers, response_tokens)
    ]


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        return "".join(_completion_to_text(part) for part in completion)
    return str(completion)
