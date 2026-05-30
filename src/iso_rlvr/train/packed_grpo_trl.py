from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from iso_rlvr.eval.packed_diagnostics import diagnose_packed_parse
from iso_rlvr.eval.run_packed_eval import think_block_diagnostics
from iso_rlvr.eval.run_packed_rollout_audit import summarize_rollout_records
from iso_rlvr.io import load_yaml, read_jsonl
from iso_rlvr.modeling import count_completion_tokens, load_causal_lm
from iso_rlvr.rewards.packed_iso import PackedRewardConfig, score_packed_completion
from iso_rlvr.train.packed_grpo_lite import evaluate_greedy


def parse_family_type_filter(value: Any) -> set[str] | None:
    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    family_types: set[str] = set()
    for item in values:
        family_types.update(part.strip() for part in str(item).split(",") if part.strip())
    return family_types or None


def filter_rows_by_family_type(
    rows: list[dict[str, Any]],
    include_family_types: set[str] | None,
) -> list[dict[str, Any]]:
    if include_family_types is None:
        return rows
    filtered = [row for row in rows if str(row.get("family_type", "")) in include_family_types]
    if not filtered:
        raise ValueError(
            f"Family type filter {sorted(include_family_types)} removed all rows."
        )
    return filtered


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


def make_packed_trl_reward_func(
    cfg: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
    log_path: Path | None = None,
):
    cfg = cfg or {}
    reward_cfg = packed_trl_reward_config(cfg)
    samples_per_prompt = int(cfg.get("num_generations", cfg.get("samples_per_prompt", 1)))
    reward_std_threshold = float(cfg.get("reward_std_threshold", 0.05))
    parse_complete_threshold = float(cfg.get("parse_complete_threshold", 0.95))

    def reward_func(
        prompts: Sequence[str],
        completions: Sequence[Any],
        gold_answers: Sequence[Sequence[str]],
        family_id: Sequence[str],
        family_type: Sequence[str],
        variant_ids: Sequence[Sequence[str]],
        num_variants: Sequence[int],
        completion_ids: Sequence[Sequence[int]] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards, records = packed_trl_reward_records(
            prompts=prompts,
            completions=completions,
            gold_answers=gold_answers,
            family_id=family_id,
            family_type=family_type,
            variant_ids=variant_ids,
            num_variants=num_variants,
            tokenizer=tokenizer,
            completion_ids=completion_ids,
            config=reward_cfg,
            extra_columns=kwargs,
        )
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    _json_dumps(
                        {
                            "summary": _summarize_reward_records(
                                records,
                                samples_per_prompt=samples_per_prompt,
                                reward_std_threshold=reward_std_threshold,
                                parse_complete_threshold=parse_complete_threshold,
                            ),
                            "records": records,
                        }
                    )
                )
                handle.write("\n")
        return rewards

    reward_func.__name__ = "packed_xml_reward"
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

    response_tokens = _response_token_counts(completions, completion_ids, tokenizer=None)
    return [
        score_packed_completion(
            _completion_to_text(completion),
            answers,
            response_tokens=tokens,
            config=config,
        ).reward
        for completion, answers, tokens in zip(completions, gold_answers, response_tokens)
    ]


def packed_trl_reward_records(
    prompts: Sequence[str],
    completions: Sequence[Any],
    gold_answers: Sequence[Sequence[str]],
    family_id: Sequence[str],
    family_type: Sequence[str],
    variant_ids: Sequence[Sequence[str]],
    num_variants: Sequence[int],
    tokenizer: Any | None = None,
    completion_ids: Sequence[Sequence[int]] | None = None,
    config: PackedRewardConfig | None = None,
    extra_columns: dict[str, Any] | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    _require_matching_lengths(
        completions=completions,
        prompts=prompts,
        gold_answers=gold_answers,
        family_id=family_id,
        family_type=family_type,
        variant_ids=variant_ids,
        num_variants=num_variants,
    )
    if completion_ids is not None and len(completion_ids) != len(completions):
        raise ValueError("completion_ids must match completions length.")

    response_tokens = _response_token_counts(completions, completion_ids, tokenizer)
    extra_columns = extra_columns or {}
    records = []
    rewards = []
    for idx, completion in enumerate(completions):
        text = _completion_to_text(completion)
        think_diagnostics = think_block_diagnostics(text)
        scored = score_packed_completion(
            text,
            gold_answers[idx],
            response_tokens=response_tokens[idx],
            config=config,
        )
        diagnostics = diagnose_packed_parse(scored.parse, gold_answers[idx])
        record = {
            "all_family_correct": scored.all_family_correct,
            "correctness": scored.correctness,
            "diagnostics": {
                "answer_count_mismatch": diagnostics.answer_count_mismatch,
                "copied_answer_indices": diagnostics.copied_answer_indices,
                "extra_answer_count": diagnostics.extra_answer_count,
                "missing_indices": diagnostics.missing_indices,
                "only_first_answer": diagnostics.only_first_answer,
                "repeated_answer": diagnostics.repeated_answer,
                "same_wrong_additive_offset": diagnostics.same_wrong_additive_offset,
                "same_wrong_multiplicative_offset": diagnostics.same_wrong_multiplicative_offset,
                "suspicious": diagnostics.suspicious,
            },
            "extra_answer_penalty": scored.extra_answer_penalty,
            "extra_answers": scored.parse.extra_answers,
            "family_component": scored.family_component,
            "family_id": family_id[idx],
            "family_mean": scored.family_mean,
            "family_type": family_type[idx],
            "format_component": scored.format_component,
            "gold_answers": list(gold_answers[idx]),
            "has_think_block": think_diagnostics["has_think_block"],
            "length_penalty": scored.length_penalty,
            "metadata": _column_value(extra_columns, "metadata", idx, []),
            "missing_indices": scored.parse.missing_indices,
            "model_response": text,
            "nontrivial_think_block": think_diagnostics["nontrivial_think_block"],
            "num_variants": num_variants[idx],
            "parse_complete": scored.parse.complete,
            "parse_mode": scored.parse.mode,
            "parsed_answers": scored.parse.answers,
            "problems": _column_value(extra_columns, "problems", idx, []),
            "prompt": prompts[idx],
            "prompt_format": _column_value(extra_columns, "prompt_format", idx, "xml"),
            "response_tokens": response_tokens[idx],
            "reward": scored.reward,
            "sample_idx": idx,
            "variant_ids": list(variant_ids[idx]),
        }
        rewards.append(float(record["reward"]))
        records.append(record)
    return rewards, records


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        return "".join(_completion_to_text(part) for part in completion)
    return str(completion)


def _response_token_counts(
    completions: Sequence[Any],
    completion_ids: Sequence[Sequence[int]] | None,
    tokenizer: Any | None,
) -> list[int | None]:
    if completion_ids is not None:
        return [len(ids) for ids in completion_ids]
    if tokenizer is None:
        return [None] * len(completions)
    return [
        count_completion_tokens(tokenizer, "", _completion_to_text(completion))
        for completion in completions
    ]


def _require_matching_lengths(**columns: Sequence[Any]) -> None:
    lengths = {name: len(value) for name, value in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"TRL reward columns must have matching lengths: {lengths}")


def _column_value(columns: dict[str, Any], key: str, idx: int, default: Any) -> Any:
    values = columns.get(key)
    if values is None:
        return default
    return values[idx]


def _summarize_reward_records(
    records: list[dict[str, Any]],
    samples_per_prompt: int,
    reward_std_threshold: float,
    parse_complete_threshold: float,
) -> dict[str, Any]:
    summary = summarize_rollout_records(
        records,
        samples_per_prompt=samples_per_prompt,
        reward_std_threshold=reward_std_threshold,
        parse_complete_threshold=parse_complete_threshold,
    )
    prompt_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        prompt_groups.setdefault(str(record["family_id"]), []).append(record)
    summary["per_prompt_rewards"] = [
        {
            "family_id": family_id,
            "family_type": group[0]["family_type"],
            "rewards": [record["reward"] for record in group],
            "all_family_correct": [record["all_family_correct"] for record in group],
            "parsed_answers": [record["parsed_answers"] for record in group],
        }
        for family_id, group in sorted(prompt_groups.items())
    ]
    return summary


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True)


def train(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model, _loaded_tokenizer, _device = load_causal_lm(cfg["model_name"], cfg.get("device", "auto"))
    model = PeftModel.from_pretrained(model, cfg["adapter_path"], is_trainable=True)

    train_rows = read_jsonl(cfg["train_dataset"])
    train_rows = filter_rows_by_family_type(
        train_rows,
        parse_family_type_filter(cfg.get("include_train_family_types")),
    )
    if cfg.get("max_train_examples"):
        train_rows = train_rows[: int(cfg["max_train_examples"])]
    eval_rows = read_jsonl(cfg["eval_dataset"])
    eval_rows = filter_rows_by_family_type(
        eval_rows,
        parse_family_type_filter(cfg.get("include_eval_family_types")),
    )
    if cfg.get("max_eval_examples"):
        eval_rows = eval_rows[: int(cfg["max_eval_examples"])]

    reward_log_path = output_dir / "reward_calls.jsonl"
    if reward_log_path.exists():
        reward_log_path.unlink()
    reward_func = make_packed_trl_reward_func(
        cfg,
        tokenizer=tokenizer,
        log_path=reward_log_path,
    )
    args = GRPOConfig(
        output_dir=str(output_dir / "trainer"),
        max_steps=int(cfg.get("max_steps", 10)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(cfg.get("learning_rate", 1e-6)),
        logging_steps=int(cfg.get("logging_steps", 1)),
        save_steps=int(cfg.get("save_steps", 10)),
        save_strategy=str(cfg.get("save_strategy", "steps")),
        save_total_limit=int(cfg.get("save_total_limit", 1)),
        report_to="none",
        remove_unused_columns=False,
        bf16=bool(cfg.get("bf16", True)),
        max_prompt_length=int(cfg.get("max_prompt_length", 512)),
        max_completion_length=int(cfg.get("max_completion_length", 256)),
        num_generations=int(cfg.get("num_generations", 4)),
        temperature=float(cfg.get("temperature", 0.7)),
        top_p=float(cfg.get("top_p", 1.0)),
        top_k=None,
        beta=float(cfg.get("beta", 0.04)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        log_completions=bool(cfg.get("log_completions", False)),
        scale_rewards=bool(cfg.get("scale_rewards", True)),
        seed=int(cfg.get("seed", 42)),
        data_seed=int(cfg.get("seed", 42)),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=args,
        train_dataset=Dataset.from_list(train_rows),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "adapter_or_model"))
    tokenizer.save_pretrained(output_dir / "adapter_or_model")

    reward_cfg = packed_trl_reward_config(cfg)
    eval_cfg = dict(cfg)
    eval_cfg.setdefault("max_new_tokens", int(eval_cfg.get("max_completion_length", 256)))
    final_summary, final_records = evaluate_greedy(
        model,
        tokenizer,
        eval_rows,
        reward_cfg,
        eval_cfg,
    )
    (output_dir / "final_eval.summary.json").write_text(
        _json_dumps(final_summary),
        encoding="utf-8",
    )
    with (output_dir / "final_eval.jsonl").open("w", encoding="utf-8") as handle:
        for record in final_records:
            handle.write(_json_dumps(record) + "\n")
    print(_json_dumps(final_summary))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
