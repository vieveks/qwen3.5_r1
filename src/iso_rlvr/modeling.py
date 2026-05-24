from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_causal_lm(model_name: str, device: str = "auto") -> tuple[Any, Any, str]:
    resolved_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if resolved_device == "cuda" else None,
        trust_remote_code=True,
    )
    if resolved_device != "cuda":
        model = model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device


def count_completion_tokens(tokenizer: Any, prompt: str, response: str) -> int:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + response, add_special_tokens=False)["input_ids"]
    return max(len(full_ids) - len(prompt_ids), 0)

