from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

from peft import PeftModel
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm import tqdm

from iso_rlvr.eval.packed_diagnostics import diagnose_packed_parse
from iso_rlvr.eval.run_eval import generate_one
from iso_rlvr.eval.run_packed_eval import build_generation_prompt
from iso_rlvr.eval.run_packed_rollout_audit import summarize_rollout_records
from iso_rlvr.io import load_yaml, read_jsonl
from iso_rlvr.modeling import count_completion_tokens, load_causal_lm
from iso_rlvr.rewards.packed_iso import PackedRewardConfig, score_packed_completion
from iso_rlvr.train.grpo_lite import sequence_logprobs


def packed_reward_config(cfg: dict[str, Any]) -> PackedRewardConfig:
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
        token_cap=int(cfg.get("token_cap", cfg.get("max_new_tokens", 256))),
    )


def grouped_advantages(
    rewards: torch.Tensor,
    group_ids: list[str],
    eps: float = 1e-6,
) -> torch.Tensor:
    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have the same length.")

    advantages = torch.zeros_like(rewards)
    for group_id in sorted(set(group_ids)):
        indices = [idx for idx, value in enumerate(group_ids) if value == group_id]
        group_rewards = rewards[indices]
        std = group_rewards.std(unbiased=False)
        if float(std) <= eps:
            continue
        advantages[indices] = (group_rewards - group_rewards.mean()) / std.clamp_min(eps)
    return advantages


def packed_grpo_loss(
    model,
    tokenizer,
    sequences: list[torch.Tensor],
    prompt_lengths: list[int],
    advantages: torch.Tensor,
) -> torch.Tensor:
    padded = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id).to(
        model.device
    )
    attention_mask = (padded != tokenizer.pad_token_id).long()
    token_log_probs = sequence_logprobs(model, padded, attention_mask)
    advantages = advantages.to(model.device)

    losses = []
    for idx, prompt_len in enumerate(prompt_lengths):
        start = max(prompt_len - 1, 0)
        end = attention_mask[idx, 1:].sum()
        completion_logprob = token_log_probs[idx, start:end].mean()
        losses.append(-advantages[idx] * completion_logprob)
    return torch.stack(losses).mean()


def sample_completion(model, tokenizer, prompt: str, cfg: dict[str, Any]) -> tuple[str, torch.Tensor]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(cfg["max_new_tokens"]),
            do_sample=True,
            temperature=float(cfg.get("temperature", 0.7)),
            top_p=float(cfg.get("top_p", 1.0)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if was_training:
        model.train()
    completion_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return response, outputs[0].detach().cpu()


def score_rollout(
    row: dict[str, Any],
    sample_idx: int,
    prompt: str,
    response: str,
    tokenizer,
    reward_cfg: PackedRewardConfig,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    response_prefix = str(cfg.get("response_prefix", ""))
    response_tokens = count_completion_tokens(tokenizer, prompt, response)
    parsed_response = response_prefix + response
    scored = score_packed_completion(
        parsed_response,
        row["gold_answers"],
        response_tokens=response_tokens,
        config=reward_cfg,
    )
    diagnostics = diagnose_packed_parse(scored.parse, row["gold_answers"])
    return {
        **row,
        "sample_idx": sample_idx,
        "response_prefix": response_prefix,
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


def build_rollout_batch(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    reward_cfg: PackedRewardConfig,
    cfg: dict[str, Any],
) -> tuple[list[torch.Tensor], list[int], torch.Tensor, list[str], list[dict[str, Any]]]:
    samples_per_prompt = int(cfg.get("samples_per_prompt", 4))
    sequences = []
    prompt_lengths = []
    group_ids = []
    records = []
    for row in rows:
        prompt = build_generation_prompt(tokenizer, row["prompt"], cfg)
        prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for sample_idx in range(samples_per_prompt):
            response, sequence = sample_completion(model, tokenizer, prompt, cfg)
            sequences.append(sequence)
            prompt_lengths.append(prompt_len)
            group_ids.append(str(row["family_id"]))
            records.append(
                score_rollout(
                    row,
                    sample_idx=sample_idx,
                    prompt=prompt,
                    response=response,
                    tokenizer=tokenizer,
                    reward_cfg=reward_cfg,
                    cfg=cfg,
                )
            )
    rewards = torch.tensor([float(record["reward"]) for record in records], dtype=torch.float32)
    return sequences, prompt_lengths, rewards, group_ids, records


def evaluate_greedy(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    reward_cfg: PackedRewardConfig,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = []
    eval_cfg = dict(cfg)
    eval_cfg["temperature"] = 0.0
    for row in tqdm(rows, desc="packed-grpo-eval", leave=False):
        prompt = build_generation_prompt(tokenizer, row["prompt"], eval_cfg)
        response = generate_one(model, tokenizer, prompt, eval_cfg)
        records.append(
            score_rollout(
                row,
                sample_idx=0,
                prompt=prompt,
                response=response,
                tokenizer=tokenizer,
                reward_cfg=reward_cfg,
                cfg=eval_cfg,
            )
        )
    summary = summarize_rollout_records(
        records,
        samples_per_prompt=1,
        reward_std_threshold=float(cfg.get("reward_std_threshold", 0.05)),
        parse_complete_threshold=float(cfg.get("parse_complete_threshold", 0.85)),
    )
    return summary, records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def train(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 13))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(cfg["dataset_path"])
    if cfg.get("max_train_examples"):
        train_rows = train_rows[: int(cfg["max_train_examples"])]
    eval_rows = read_jsonl(cfg["eval_dataset_path"])
    if cfg.get("max_eval_examples"):
        eval_rows = eval_rows[: int(cfg["max_eval_examples"])]

    model, tokenizer, _device = load_causal_lm(cfg["model_name"], cfg.get("device", "auto"))
    if cfg.get("adapter_path"):
        model = PeftModel.from_pretrained(model, cfg["adapter_path"], is_trainable=True)
    else:
        raise ValueError("packed_grpo_lite requires adapter_path for this Phase 5 smoke.")
    model.train()
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(cfg["learning_rate"]),
    )

    reward_cfg = packed_reward_config(cfg)
    rng = random.Random(seed)
    max_steps = int(cfg["max_steps"])
    prompts_per_step = int(cfg.get("prompts_per_step", 1))
    eval_every_steps = int(cfg.get("eval_every_steps", 10))

    log_path = output_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_handle:
        for step in tqdm(range(max_steps), desc="packed-grpo-lite"):
            batch_rows = rng.sample(train_rows, prompts_per_step)
            sequences, prompt_lengths, rewards, group_ids, records = build_rollout_batch(
                model,
                tokenizer,
                batch_rows,
                reward_cfg,
                cfg,
            )
            advantages = grouped_advantages(rewards, group_ids)
            loss = packed_grpo_loss(model, tokenizer, sequences, prompt_lengths, advantages)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            summary = summarize_rollout_records(
                records,
                samples_per_prompt=int(cfg.get("samples_per_prompt", 4)),
                reward_std_threshold=float(cfg.get("reward_std_threshold", 0.05)),
                parse_complete_threshold=float(cfg.get("parse_complete_threshold", 0.85)),
            )
            metrics = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std(unbiased=False)),
                "advantage_std": float(advantages.std(unbiased=False)),
                **summary,
            }
            log_record: dict[str, Any] = {"metrics": metrics, "records": records}

            if eval_every_steps > 0 and (step + 1) % eval_every_steps == 0:
                eval_summary, eval_records = evaluate_greedy(
                    model,
                    tokenizer,
                    eval_rows,
                    reward_cfg,
                    cfg,
                )
                eval_path = output_dir / f"eval_step_{step + 1:04d}.jsonl"
                write_jsonl(eval_path, eval_records)
                (output_dir / f"eval_step_{step + 1:04d}.summary.json").write_text(
                    json.dumps(eval_summary, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                log_record["eval_summary"] = eval_summary

            log_handle.write(json.dumps(log_record, sort_keys=True) + "\n")
            log_handle.flush()

    final_summary, final_records = evaluate_greedy(model, tokenizer, eval_rows, reward_cfg, cfg)
    write_jsonl(output_dir / "final_eval.jsonl", final_records)
    (output_dir / "final_eval.summary.json").write_text(
        json.dumps(final_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    model.save_pretrained(output_dir / "adapter_or_model")
    tokenizer.save_pretrained(output_dir / "adapter_or_model")
    print(json.dumps(final_summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
