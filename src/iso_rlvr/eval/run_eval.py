from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm

from iso_rlvr.io import load_yaml, read_jsonl
from iso_rlvr.modeling import count_completion_tokens, load_causal_lm
from iso_rlvr.rewards.answer import extract_answer, is_correct
from iso_rlvr.rewards.iso import ScoredResponse, summarize
from iso_rlvr.eval.summary import summarize_by_family_type


def generate_one(model, tokenizer, prompt: str, cfg: dict) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = float(cfg.get("temperature", 0.0)) > 0.0
    generation_kwargs = {
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(cfg.get("temperature", 1.0))
        generation_kwargs["top_p"] = float(cfg.get("top_p", 1.0))
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    completion_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def run_eval(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["dataset_path"])
    if cfg.get("max_examples"):
        rows = rows[: int(cfg["max_examples"])]

    output_path = Path(cfg["output_path"])
    resume = bool(cfg.get("resume", False))
    outputs = read_jsonl(output_path) if resume and output_path.exists() else []
    completed = {(row["family_id"], row["variant_id"]) for row in outputs}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer, _device = load_causal_lm(cfg["model_name"], cfg.get("device", "auto"))
    if cfg.get("adapter_path"):
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
        model.eval()
    template = cfg["prompt_template"]

    log_mode = "a" if resume and output_path.exists() else "w"
    with output_path.open(log_mode, encoding="utf-8") as output_handle:
        for row in tqdm(rows, desc="eval"):
            row_key = (row["family_id"], row["variant_id"])
            if row_key in completed:
                continue
            prompt = template.format(problem=row["problem"])
            response = generate_one(model, tokenizer, prompt, cfg)
            extracted = extract_answer(response)
            correct = is_correct(extracted, row["answer"])
            token_count = count_completion_tokens(tokenizer, prompt, response)
            result = {
                **row,
                "prompt": prompt,
                "model_response": response,
                "extracted_answer": extracted,
                "correct": correct,
                "response_tokens": token_count,
            }
            outputs.append(result)
            completed.add(row_key)
            output_handle.write(json.dumps(result, sort_keys=True) + "\n")
            output_handle.flush()

    scored = []
    for row in outputs:
        scored.append(
            ScoredResponse(
                family_id=row["family_id"],
                variant_id=row["variant_id"],
                gold=row["answer"],
                response=row["model_response"],
                correct=row["correct"],
                extracted_answer=row["extracted_answer"],
                token_count=row["response_tokens"],
            )
        )

    summary = summarize(scored)
    summary["by_family_type"] = summarize_by_family_type(outputs)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_eval(args.config)


if __name__ == "__main__":
    main()
