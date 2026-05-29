from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from peft import PeftModel
from tqdm import tqdm

from iso_rlvr.eval.packed_diagnostics import diagnose_packed_parse
from iso_rlvr.eval.run_eval import generate_one
from iso_rlvr.io import load_yaml, read_jsonl
from iso_rlvr.modeling import count_completion_tokens, load_causal_lm
from iso_rlvr.rewards.packed_iso import score_packed_completion


THINK_BLOCK_PATTERN = re.compile(
    r"<think\b[^>]*>(.*?)</think>",
    re.IGNORECASE | re.DOTALL,
)
MINIMAL_THINK_ANCHOR = "Let me solve this."


def think_block_diagnostics(completion: str) -> dict[str, Any]:
    matches = list(THINK_BLOCK_PATTERN.finditer(completion))
    if not matches:
        return {
            "has_think_block": False,
            "nontrivial_think_block": False,
            "think_text": "",
        }

    think_text = matches[-1].group(1).strip()
    normalized_think = _normalize_think_text(think_text)
    return {
        "has_think_block": True,
        "nontrivial_think_block": bool(normalized_think)
        and normalized_think != _normalize_think_text(MINIMAL_THINK_ANCHOR),
        "think_text": think_text,
    }


def _normalize_think_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def build_generation_prompt(tokenizer: Any, row_prompt: str, cfg: dict[str, Any]) -> str:
    response_prefix = str(cfg.get("response_prefix", ""))
    if not bool(cfg.get("apply_chat_template", False)):
        return row_prompt + response_prefix

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("apply_chat_template=true requires a tokenizer with chat template support")

    messages = []
    system_prompt = cfg.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": row_prompt})

    try:
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise ValueError(
            "apply_chat_template=true failed; check that the model tokenizer defines a chat_template"
        ) from exc
    return rendered_prompt + response_prefix


def summarize_packed_eval_rows(
    rows: list[dict[str, Any]],
    include_by_family_type: bool = True,
) -> dict[str, Any]:
    if not rows:
        summary: dict[str, Any] = {
            "examples": 0,
            "variant_examples": 0,
            "accuracy": 0.0,
            "family_accuracy": 0.0,
            "parse_complete_rate": 0.0,
            "avg_reward": 0.0,
            "answer_count_mismatch_rate": 0.0,
            "suspicious_rate": 0.0,
            "think_block_rate": 0.0,
            "nontrivial_think_block_rate": 0.0,
        }
        if include_by_family_type:
            summary["by_family_type"] = {}
        return summary

    total_variants = sum(len(row["gold_answers"]) for row in rows)
    correct_variants = sum(sum(row["correctness"]) for row in rows)
    summary = {
        "examples": len(rows),
        "variant_examples": total_variants,
        "accuracy": correct_variants / total_variants if total_variants else 0.0,
        "family_accuracy": sum(row["all_family_correct"] for row in rows) / len(rows),
        "parse_complete_rate": sum(row["parse_complete"] for row in rows) / len(rows),
        "avg_reward": sum(float(row["reward"]) for row in rows) / len(rows),
        "answer_count_mismatch_rate": sum(
            row["diagnostics"]["answer_count_mismatch"] for row in rows
        )
        / len(rows),
        "suspicious_rate": sum(row["diagnostics"]["suspicious"] for row in rows) / len(rows),
        "think_block_rate": sum(row.get("has_think_block", False) for row in rows) / len(rows),
        "nontrivial_think_block_rate": sum(
            row.get("nontrivial_think_block", False) for row in rows
        )
        / len(rows),
    }
    if include_by_family_type:
        by_type: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_type.setdefault(str(row["family_type"]), []).append(row)
        summary["by_family_type"] = {
            family_type: summarize_packed_eval_rows(
                type_rows,
                include_by_family_type=False,
            )
            for family_type, type_rows in sorted(by_type.items())
        }
    return summary


def run_packed_eval(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["dataset_path"])
    if cfg.get("max_examples"):
        rows = rows[: int(cfg["max_examples"])]

    output_path = Path(cfg["output_path"])
    resume = bool(cfg.get("resume", False))
    outputs = read_jsonl(output_path) if resume and output_path.exists() else []
    completed = {row["family_id"] for row in outputs}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer, _device = load_causal_lm(cfg["model_name"], cfg.get("device", "auto"))
    if cfg.get("adapter_path"):
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
        model.eval()

    log_mode = "a" if resume and output_path.exists() else "w"
    with output_path.open(log_mode, encoding="utf-8") as output_handle:
        for row in tqdm(rows, desc="packed-eval"):
            if row["family_id"] in completed:
                continue
            response_prefix = str(cfg.get("response_prefix", ""))
            prompt = build_generation_prompt(tokenizer, row["prompt"], cfg)
            response = generate_one(model, tokenizer, prompt, cfg)
            response_tokens = count_completion_tokens(tokenizer, prompt, response)
            parsed_response = response_prefix + response
            scored = score_packed_completion(
                parsed_response,
                row["gold_answers"],
                response_tokens=response_tokens,
            )
            diagnostics = diagnose_packed_parse(scored.parse, row["gold_answers"])
            think_diagnostics = think_block_diagnostics(parsed_response)
            result = {
                **row,
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
            outputs.append(result)
            completed.add(row["family_id"])
            output_handle.write(json.dumps(result, sort_keys=True) + "\n")
            output_handle.flush()

    summary = summarize_packed_eval_rows(outputs)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_packed_eval(args.config)


if __name__ == "__main__":
    main()
