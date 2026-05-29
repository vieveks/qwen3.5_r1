from iso_rlvr.eval.run_packed_eval import (
    build_generation_prompt,
    summarize_packed_eval_rows,
    think_block_diagnostics,
)
from iso_rlvr.rewards.packed_iso import score_packed_completion


def _row(
    family_type: str,
    correctness: list[bool],
    all_family_correct: bool,
    reward: float,
    parse_complete: bool = True,
    answer_count_mismatch: bool = False,
    suspicious: bool = False,
) -> dict:
    return {
        "family_type": family_type,
        "gold_answers": ["1"] * len(correctness),
        "correctness": correctness,
        "all_family_correct": all_family_correct,
        "reward": reward,
        "parse_complete": parse_complete,
        "diagnostics": {
            "answer_count_mismatch": answer_count_mismatch,
            "suspicious": suspicious,
        },
    }


def test_summarize_packed_eval_rows_computes_overall_and_type_metrics():
    summary = summarize_packed_eval_rows(
        [
            {
                **_row("missing_average", [True, True], True, 1.55),
                "has_think_block": True,
                "nontrivial_think_block": False,
            },
            _row(
                "rational_linear_equation",
                [True, False],
                False,
                0.61,
                parse_complete=False,
                answer_count_mismatch=True,
                suspicious=True,
            ),
        ]
    )

    assert summary["examples"] == 2
    assert summary["variant_examples"] == 4
    assert summary["accuracy"] == 0.75
    assert summary["family_accuracy"] == 0.5
    assert summary["parse_complete_rate"] == 0.5
    assert summary["avg_reward"] == 1.08
    assert summary["answer_count_mismatch_rate"] == 0.5
    assert summary["suspicious_rate"] == 0.5
    assert summary["think_block_rate"] == 0.5
    assert summary["nontrivial_think_block_rate"] == 0.0
    assert summary["by_family_type"]["missing_average"]["accuracy"] == 1.0
    assert summary["by_family_type"]["rational_linear_equation"]["accuracy"] == 0.5
    assert "by_family_type" not in summary["by_family_type"]["missing_average"]


def test_summarize_packed_eval_rows_handles_empty_input():
    summary = summarize_packed_eval_rows([])

    assert summary["examples"] == 0
    assert summary["by_family_type"] == {}


def test_response_prefix_can_be_combined_with_generated_suffix_for_scoring():
    response_prefix = "\nAnswer 1:"
    generated_suffix = " 3\nAnswer 2: 7"

    scored = score_packed_completion(response_prefix + generated_suffix, ["3", "7"])

    assert scored.parse.answers == ["3", "7"]
    assert scored.parse.complete
    assert scored.reward == 1.55


def test_think_block_diagnostics_identifies_anchor_and_nontrivial_text():
    anchor = think_block_diagnostics("<think>\nLet me solve this.\n</think>")
    nontrivial = think_block_diagnostics(
        "<think>\nLet me solve this. Then I compute x = 3.\n</think>"
    )
    missing = think_block_diagnostics("<answers></answers>")

    assert anchor["has_think_block"]
    assert not anchor["nontrivial_think_block"]
    assert nontrivial["has_think_block"]
    assert nontrivial["nontrivial_think_block"]
    assert not missing["has_think_block"]
    assert not missing["nontrivial_think_block"]


class _ChatTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "<chat>" + messages[-1]["content"] + "<assistant>"


def test_build_generation_prompt_can_render_chat_template_with_system_prompt():
    tokenizer = _ChatTokenizer()
    prompt = build_generation_prompt(
        tokenizer,
        "Problem 1: 1 + 2\n\nAnswer 1: <number>",
        {
            "apply_chat_template": True,
            "system_prompt": "Return only final answers.",
            "response_prefix": "\nAnswer 1:",
        },
    )

    assert tokenizer.messages == [
        {"role": "system", "content": "Return only final answers."},
        {"role": "user", "content": "Problem 1: 1 + 2\n\nAnswer 1: <number>"},
    ]
    assert prompt.endswith("<assistant>\nAnswer 1:")


def test_build_generation_prompt_defaults_to_plain_prompt():
    prompt = build_generation_prompt(
        object(),
        "Problem 1: 1 + 2",
        {"response_prefix": "\nAnswer 1:"},
    )

    assert prompt == "Problem 1: 1 + 2\nAnswer 1:"
