from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm import tqdm

from iso_rlvr.io import group_by_family, load_yaml, read_jsonl
from iso_rlvr.modeling import count_completion_tokens, load_causal_lm
from iso_rlvr.rewards.answer import extract_answer, is_correct
from iso_rlvr.rewards.iso import ScoredResponse, iso_reward_values, summarize


def maybe_add_lora(model, cfg: dict):
    lora_cfg = cfg.get("lora", {})
    if not lora_cfg.get("enabled", False):
        return model
    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def sample_completion(model, tokenizer, prompt: str, cfg: dict) -> tuple[str, torch.Tensor]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(cfg["max_new_tokens"]),
            do_sample=True,
            temperature=float(cfg.get("temperature", 1.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return response, outputs[0].detach().cpu()


def sequence_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs


def grpo_loss(
    model,
    tokenizer,
    sequences: list[torch.Tensor],
    prompt_lengths: list[int],
    rewards: torch.Tensor,
) -> torch.Tensor:
    padded = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id).to(
        model.device
    )
    attention_mask = (padded != tokenizer.pad_token_id).long()
    token_log_probs = sequence_logprobs(model, padded, attention_mask)

    losses = []
    centered = rewards - rewards.mean()
    denom = rewards.std(unbiased=False).clamp_min(1e-6)
    advantages = (centered / denom).to(model.device)

    for idx, prompt_len in enumerate(prompt_lengths):
        # token_log_probs is shifted by one; completion starts at prompt_len - 1.
        start = max(prompt_len - 1, 0)
        end = attention_mask[idx, 1:].sum()
        completion_logprob = token_log_probs[idx, start:end].mean()
        losses.append(-advantages[idx] * completion_logprob)
    return torch.stack(losses).mean()


def reward_values(scored: list[ScoredResponse], cfg: dict) -> list[float]:
    reward_mode = cfg.get("reward_mode", "iso")
    if reward_mode == "independent":
        return [float(item.correct) for item in scored]
    if reward_mode == "iso":
        return iso_reward_values(scored, float(cfg.get("lambda_iso", 0.5)))
    raise ValueError(f"Unknown reward_mode {reward_mode!r}; expected 'independent' or 'iso'")


def build_rollout_batch(model, tokenizer, families: list[list[dict]], cfg: dict):
    template = cfg["prompt_template"]
    sequences = []
    prompt_lengths = []
    scored = []
    records = []
    for family in families:
        for row in family[: int(cfg["variants_per_family"])]:
            prompt = template.format(problem=row["problem"])
            for _ in range(int(cfg.get("samples_per_variant", 1))):
                response, sequence = sample_completion(model, tokenizer, prompt, cfg)
                extracted = extract_answer(response)
                correct = is_correct(extracted, row["answer"])
                token_count = count_completion_tokens(tokenizer, prompt, response)
                prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
                sequences.append(sequence)
                prompt_lengths.append(prompt_len)
                scored.append(
                    ScoredResponse(
                        family_id=row["family_id"],
                        variant_id=row["variant_id"],
                        gold=row["answer"],
                        response=response,
                        correct=correct,
                        extracted_answer=extracted,
                        token_count=token_count,
                    )
                )
                records.append(
                    {
                        "family_id": row["family_id"],
                        "variant_id": row["variant_id"],
                        "problem": row["problem"],
                        "answer": row["answer"],
                        "response": response,
                        "extracted_answer": extracted,
                        "correct": correct,
                        "response_tokens": token_count,
                    }
                )
    rewards = torch.tensor(reward_values(scored, cfg), dtype=torch.float32)
    return sequences, prompt_lengths, rewards, scored, records


def train(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 13))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(cfg["dataset_path"])
    grouped = list(group_by_family(rows).values())
    rng = random.Random(seed)

    model, tokenizer, _device = load_causal_lm(cfg["model_name"], cfg.get("device", "auto"))
    model.train()
    model = maybe_add_lora(model, cfg)
    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]))

    log_path = output_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_handle:
        for step in tqdm(range(int(cfg["max_steps"])), desc="train"):
            batch_families = rng.sample(grouped, int(cfg["families_per_step"]))
            sequences, prompt_lengths, rewards, scored, records = build_rollout_batch(
                model, tokenizer, batch_families, cfg
            )
            loss = grpo_loss(model, tokenizer, sequences, prompt_lengths, rewards)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            metrics = summarize(scored)
            metrics.update(
                {
                    "step": step,
                    "seed": seed,
                    "reward_mode": cfg.get("reward_mode", "iso"),
                    "loss": float(loss.detach().cpu()),
                    "mean_reward": float(rewards.mean()),
                    "max_reward": float(rewards.max()),
                    "min_reward": float(rewards.min()),
                }
            )
            log_handle.write(json.dumps({"metrics": metrics, "records": records}, sort_keys=True) + "\n")
            log_handle.flush()

    model.save_pretrained(output_dir / "adapter_or_model")
    tokenizer.save_pretrained(output_dir / "adapter_or_model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
