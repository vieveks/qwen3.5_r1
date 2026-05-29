from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any

from peft import PeftModel
from tqdm import tqdm

from iso_rlvr.eval.packed_diagnostics import diagnose_packed_parse
from iso_rlvr.eval.run_eval import generate_one
from iso_rlvr.eval.run_packed_eval import build_generation_prompt, think_block_diagnostics
from iso_rlvr.io import load_yaml, read_jsonl
from iso_rlvr.modeling import count_completion_tokens, load_causal_lm
from iso_rlvr.rewards.packed_iso import PackedRewardConfig, score_packed_completion


def pure_format_correctness_reward_config(cfg: dict[str, Any]) -> PackedRewardConfig:
    return PackedRewardConfig(
        format_reward=float(cfg.get("format_reward", 0.05)),
        missing_format_penalty=float(cfg.get("missing_format_penalty", -0.10)),
        family_mean_weight=0.0,
        all_family_correct_weight=0.0,
        extra_answer_penalty=float(cfg.get("extra_answer_penalty", 0.10)),
        length_penalty_weight=float(cfg.get("length_penalty_weight", 0.05)),
        token_cap=int(cfg.get("token_cap", cfg.get("max_new_tokens", 256))),
    )


def reward_histogram(rewards: list[float], bucket_size: float = 0.25) -> dict[str, int]:
    if not rewards:
        return {}
    counts: Counter[str] = Counter()
    for reward in rewards:
        bucket_start = bucket_size * int(reward / bucket_size)
        if reward < 0 and reward % bucket_size:
            bucket_start -= bucket_size
        bucket_end = bucket_start + bucket_size
        counts[f"[{bucket_start:.2f},{bucket_end:.2f})"] += 1
    return dict(sorted(counts.items()))


def summarize_rollout_records(
    records: list[dict[str, Any]],
    samples_per_prompt: int,
    reward_std_threshold: float,
    parse_complete_threshold: float,
) -> dict[str, Any]:
    if not records:
        return {
            "samples": 0,
            "prompts": 0,
            "parse_complete_rate": 0.0,
            "answer_count_mismatch_rate": 0.0,
            "suspicious_rate": 0.0,
            "accuracy": 0.0,
            "family_accuracy": 0.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "reward_min": 0.0,
            "reward_max": 0.0,
            "reward_histogram": {},
            "contrast_prompt_count": 0,
            "has_prompt_level_contrast": False,
            "passes_audit_gate": False,
            "think_block_rate": 0.0,
            "nontrivial_think_block_rate": 0.0,
        }

    rewards = [float(record["reward"]) for record in records]
    total_variants = sum(len(record["gold_answers"]) for record in records)
    correct_variants = sum(sum(record["correctness"]) for record in records)
    prompt_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        prompt_groups.setdefault(str(record["family_id"]), []).append(record)

    contrast_prompt_ids = []
    for family_id, group in prompt_groups.items():
        family_correct_values = {bool(record["all_family_correct"]) for record in group}
        if family_correct_values == {False, True}:
            contrast_prompt_ids.append(family_id)

    parse_complete_rate = sum(record["parse_complete"] for record in records) / len(records)
    reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    has_prompt_level_contrast = bool(contrast_prompt_ids)
    passes_gate = (
        parse_complete_rate >= parse_complete_threshold
        and reward_std > reward_std_threshold
        and has_prompt_level_contrast
    )

    by_family_type: dict[str, dict[str, Any]] = {}
    for family_type in sorted({str(record["family_type"]) for record in records}):
        type_records = [record for record in records if str(record["family_type"]) == family_type]
        type_variants = sum(len(record["gold_answers"]) for record in type_records)
        type_correct = sum(sum(record["correctness"]) for record in type_records)
        by_family_type[family_type] = {
            "samples": len(type_records),
            "variant_examples": type_variants,
            "accuracy": type_correct / type_variants if type_variants else 0.0,
            "family_accuracy": sum(record["all_family_correct"] for record in type_records)
            / len(type_records),
            "parse_complete_rate": sum(record["parse_complete"] for record in type_records)
            / len(type_records),
            "answer_count_mismatch_rate": sum(
                record["diagnostics"]["answer_count_mismatch"] for record in type_records
            )
            / len(type_records),
            "suspicious_rate": sum(
                record["diagnostics"]["suspicious"] for record in type_records
            )
            / len(type_records),
            "think_block_rate": sum(
                record.get("has_think_block", False) for record in type_records
            )
            / len(type_records),
            "nontrivial_think_block_rate": sum(
                record.get("nontrivial_think_block", False) for record in type_records
            )
            / len(type_records),
        }

    return {
        "samples": len(records),
        "prompts": len(prompt_groups),
        "samples_per_prompt": samples_per_prompt,
        "variant_examples": total_variants,
        "accuracy": correct_variants / total_variants if total_variants else 0.0,
        "family_accuracy": sum(record["all_family_correct"] for record in records) / len(records),
        "parse_complete_rate": parse_complete_rate,
        "answer_count_mismatch_rate": sum(
            record["diagnostics"]["answer_count_mismatch"] for record in records
        )
        / len(records),
        "suspicious_rate": sum(record["diagnostics"]["suspicious"] for record in records)
        / len(records),
        "think_block_rate": sum(record.get("has_think_block", False) for record in records)
        / len(records),
        "nontrivial_think_block_rate": sum(
            record.get("nontrivial_think_block", False) for record in records
        )
        / len(records),
        "reward_mean": statistics.fmean(rewards),
        "reward_std": reward_std,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "reward_histogram": reward_histogram(rewards),
        "contrast_prompt_count": len(contrast_prompt_ids),
        "contrast_prompt_ids": contrast_prompt_ids,
        "has_prompt_level_contrast": has_prompt_level_contrast,
        "parse_complete_threshold": parse_complete_threshold,
        "reward_std_threshold": reward_std_threshold,
        "passes_audit_gate": passes_gate,
        "by_family_type": by_family_type,
    }


def run_packed_rollout_audit(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["dataset_path"])
    if cfg.get("max_examples"):
        rows = rows[: int(cfg["max_examples"])]

    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer, _device = load_causal_lm(cfg["model_name"], cfg.get("device", "auto"))
    if cfg.get("adapter_path"):
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
        model.eval()

    samples_per_prompt = int(cfg.get("samples_per_prompt", 4))
    reward_cfg = pure_format_correctness_reward_config(cfg)
    records = []
    malformed_samples = []
    max_malformed_samples = int(cfg.get("max_malformed_samples", 20))

    with output_path.open("w", encoding="utf-8") as output_handle:
        for row in tqdm(rows, desc="packed-rollout-audit"):
            response_prefix = str(cfg.get("response_prefix", ""))
            prompt = build_generation_prompt(tokenizer, row["prompt"], cfg)
            for sample_idx in range(samples_per_prompt):
                response = generate_one(model, tokenizer, prompt, cfg)
                response_tokens = count_completion_tokens(tokenizer, prompt, response)
                parsed_response = response_prefix + response
                scored = score_packed_completion(
                    parsed_response,
                    row["gold_answers"],
                    response_tokens=response_tokens,
                    config=reward_cfg,
                )
                diagnostics = diagnose_packed_parse(scored.parse, row["gold_answers"])
                think_diagnostics = think_block_diagnostics(parsed_response)
                result = {
                    **row,
                    "sample_idx": sample_idx,
                    "apply_chat_template": bool(cfg.get("apply_chat_template", False)),
                    "response_prefix": response_prefix,
                    "generation_prompt": prompt,
                    "model_response": response,
                    "parsed_response": parsed_response,
                    "parsed_answers": scored.parse.answers,
                    "missing_indices": scored.parse.missing_indices,
                    "extra_answers": scored.parse.extra_answers,
                    "parse_mode": scored.parse.mode,
                    "parse_complete": scored.parse.complete,
                    "correctness": scored.correctness,
                    "family_mean": scored.family_mean,
                    "all_family_correct": scored.all_family_correct,
                    "reward": scored.reward,
                    "format_component": scored.format_component,
                    "family_component": scored.family_component,
                    "extra_answer_penalty": scored.extra_answer_penalty,
                    "length_penalty": scored.length_penalty,
                    "response_tokens": response_tokens,
                    **think_diagnostics,
                    "diagnostics": {
                        "repeated_answer": diagnostics.repeated_answer,
                        "copied_answer_indices": diagnostics.copied_answer_indices,
                        "only_first_answer": diagnostics.only_first_answer,
                        "missing_indices": diagnostics.missing_indices,
                        "extra_answer_count": diagnostics.extra_answer_count,
                        "answer_count_mismatch": diagnostics.answer_count_mismatch,
                        "same_wrong_additive_offset": diagnostics.same_wrong_additive_offset,
                        "same_wrong_multiplicative_offset": diagnostics.same_wrong_multiplicative_offset,
                        "suspicious": diagnostics.suspicious,
                    },
                }
                if (
                    (not scored.parse.complete or diagnostics.answer_count_mismatch)
                    and len(malformed_samples) < max_malformed_samples
                ):
                    malformed_samples.append(
                        {
                            "family_id": row["family_id"],
                            "family_type": row["family_type"],
                            "sample_idx": sample_idx,
                            "gold_answers": row["gold_answers"],
                            "parsed_answers": scored.parse.answers,
                            "parse_mode": scored.parse.mode,
                            "missing_indices": scored.parse.missing_indices,
                            "extra_answers": scored.parse.extra_answers,
                            "diagnostics": result["diagnostics"],
                            "full_completion": response,
                            "full_parsed_response": parsed_response,
                        }
                    )
                records.append(result)
                output_handle.write(json.dumps(result, sort_keys=True) + "\n")
                output_handle.flush()

    summary = summarize_rollout_records(
        records,
        samples_per_prompt=samples_per_prompt,
        reward_std_threshold=float(cfg.get("reward_std_threshold", 0.05)),
        parse_complete_threshold=float(cfg.get("parse_complete_threshold", 0.85)),
    )
    summary["malformed_samples"] = malformed_samples
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_packed_rollout_audit(args.config)


if __name__ == "__main__":
    main()
