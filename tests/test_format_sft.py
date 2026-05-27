import torch

from iso_rlvr.train.format_sft import collate_sft_batch, encode_sft_example


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        del add_special_tokens
        ids = [ord(char) for char in text]
        return {"input_ids": ids}


def test_encode_sft_example_masks_prompt_tokens_only():
    tokenizer = _Tokenizer()

    encoded = encode_sft_example(tokenizer, "prompt", "answer", separator="\n")

    prompt_len = len("prompt\n")
    assert encoded["input_ids"].tolist() == [ord(char) for char in "prompt\nanswer"]
    assert encoded["labels"][:prompt_len].tolist() == [-100] * prompt_len
    assert encoded["labels"][prompt_len:].tolist() == [ord(char) for char in "answer"]


def test_encode_sft_example_honors_max_length():
    tokenizer = _Tokenizer()

    encoded = encode_sft_example(tokenizer, "prompt", "answer", separator="\n", max_length=8)

    assert len(encoded["input_ids"]) == 8
    assert len(encoded["labels"]) == 8


def test_collate_sft_batch_pads_inputs_and_masks_labels():
    batch = collate_sft_batch(
        [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([-100, 2, 3]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([-100, 5]),
            },
        ],
        pad_token_id=0,
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]]
