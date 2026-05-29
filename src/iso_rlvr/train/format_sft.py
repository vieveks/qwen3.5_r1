from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import torch
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm import tqdm

from iso_rlvr.io import load_yaml, read_jsonl
from iso_rlvr.modeling import load_causal_lm
from iso_rlvr.train.grpo_lite import maybe_add_lora


def encode_sft_example(
    tokenizer: Any,
    prompt: str,
    completion: str,
    separator: str = "\n",
    max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    prompt_text = prompt + separator
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt_text + completion, add_special_tokens=True)["input_ids"]
    if max_length is not None:
        full_ids = full_ids[:max_length]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_sft_batch(
    examples: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    input_ids = pad_sequence(
        [example["input_ids"] for example in examples],
        batch_first=True,
        padding_value=pad_token_id,
    )
    labels = pad_sequence(
        [example["labels"] for example in examples],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = (input_ids != pad_token_id).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def iter_batches(
    encoded_rows: list[dict[str, torch.Tensor]],
    batch_size: int,
    seed: int,
    epochs: int,
):
    rng = random.Random(seed)
    for _epoch in range(epochs):
        rows = list(encoded_rows)
        rng.shuffle(rows)
        for start in range(0, len(rows), batch_size):
            yield rows[start : start + batch_size]


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
    if cfg.get("max_examples"):
        rows = rows[: int(cfg["max_examples"])]

    model, tokenizer, _device = load_causal_lm(cfg["model_name"], cfg.get("device", "auto"))
    if cfg.get("adapter_path"):
        model = PeftModel.from_pretrained(model, cfg["adapter_path"], is_trainable=True)
    else:
        model = maybe_add_lora(model, cfg)
    model.train()

    separator = str(cfg.get("separator", "\n"))
    max_length = cfg.get("max_length")
    encoded_rows = [
        encode_sft_example(
            tokenizer,
            str(row["prompt"]),
            str(row["completion"]),
            separator=separator,
            max_length=int(max_length) if max_length else None,
        )
        for row in rows
    ]

    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    batch_size = int(cfg.get("batch_size", 1))
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1))
    max_steps = int(cfg["max_steps"])
    epochs = int(cfg.get("epochs", 1000))

    log_path = output_dir / "train_log.jsonl"
    step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        progress = tqdm(total=max_steps, desc="format-sft")
        for batch_rows in iter_batches(encoded_rows, batch_size, seed=seed, epochs=epochs):
            batch = collate_sft_batch(batch_rows, tokenizer.pad_token_id)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu())

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                metrics = {
                    "step": step,
                    "loss": running_loss,
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "examples": len(rows),
                }
                log_handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                log_handle.flush()
                running_loss = 0.0
                progress.update(1)
                if progress.n >= max_steps:
                    break
            step += 1
        progress.close()

    model.save_pretrained(output_dir / "adapter_or_model")
    tokenizer.save_pretrained(output_dir / "adapter_or_model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
