from iso_rlvr.eval.run_packed_rollout_audit import (
    filter_rows_by_family_type,
    parse_family_type_filter,
    pure_format_correctness_reward_config,
    reward_histogram,
    summarize_rollout_records,
)
from iso_rlvr.rewards.packed_iso import score_packed_completion


def _record(
    family_id: str,
    family_type: str,
    correctness: list[bool],
    reward: float,
    parse_complete: bool = True,
    answer_count_mismatch: bool = False,
    suspicious: bool = False,
) -> dict:
    return {
        "family_id": family_id,
        "family_type": family_type,
        "gold_answers": ["1"] * len(correctness),
        "correctness": correctness,
        "all_family_correct": all(correctness),
        "reward": reward,
        "parse_complete": parse_complete,
        "diagnostics": {
            "answer_count_mismatch": answer_count_mismatch,
            "suspicious": suspicious,
        },
    }


def test_pure_format_correctness_reward_config_disables_family_bonus():
    cfg = pure_format_correctness_reward_config(
        {
            "format_reward": 0.05,
            "missing_format_penalty": -0.10,
            "extra_answer_penalty": 0.10,
            "length_penalty_weight": 0.0,
        }
    )

    scored = score_packed_completion(
        "<answers>\n<answer_1>2</answer_1>\n<answer_2>3</answer_2>\n</answers>",
        ["2", "3"],
        config=cfg,
    )

    assert scored.family_component == 0.0
    assert scored.reward == 1.05


def test_reward_histogram_buckets_rewards():
    assert reward_histogram([-0.1, 0.0, 0.24, 0.25, 1.05], bucket_size=0.25) == {
        "[-0.25,0.00)": 1,
        "[0.00,0.25)": 2,
        "[0.25,0.50)": 1,
        "[1.00,1.25)": 1,
    }


def test_parse_family_type_filter_accepts_commas_and_lists():
    assert parse_family_type_filter(["missing_average,rational_linear_equation"]) == {
        "missing_average",
        "rational_linear_equation",
    }
    assert parse_family_type_filter(["missing_average", "rational_linear_equation"]) == {
        "missing_average",
        "rational_linear_equation",
    }
    assert parse_family_type_filter(None) is None


def test_filter_rows_by_family_type_keeps_requested_rows():
    rows = [
        {"family_type": "missing_average"},
        {"family_type": "chinese_remainder"},
        {"family_type": "rational_linear_equation"},
    ]

    filtered = filter_rows_by_family_type(rows, {"missing_average"})

    assert filtered == [{"family_type": "missing_average"}]


def test_summarize_rollout_records_requires_prompt_level_contrast_for_gate():
    summary = summarize_rollout_records(
        [
            {
                **_record("fam_1", "missing_average", [True, True], 1.05),
                "has_think_block": True,
                "nontrivial_think_block": False,
            },
            _record("fam_1", "missing_average", [True, False], 0.55),
            _record("fam_2", "rational_linear_equation", [False, False], 0.05),
            _record("fam_2", "rational_linear_equation", [False, False], 0.05),
        ],
        samples_per_prompt=2,
        reward_std_threshold=0.05,
        parse_complete_threshold=0.85,
    )

    assert summary["samples"] == 4
    assert summary["prompts"] == 2
    assert summary["accuracy"] == 0.375
    assert summary["parse_complete_rate"] == 1.0
    assert summary["reward_std"] > 0.05
    assert summary["contrast_prompt_count"] == 1
    assert summary["contrast_prompt_ids"] == ["fam_1"]
    assert summary["passes_audit_gate"]
    assert summary["think_block_rate"] == 0.25
    assert summary["nontrivial_think_block_rate"] == 0.0
    assert summary["by_family_type"]["missing_average"]["accuracy"] == 0.75
    assert summary["by_family_type"]["missing_average"]["think_block_rate"] == 0.5


def test_summarize_rollout_records_fails_without_prompt_level_contrast():
    summary = summarize_rollout_records(
        [
            _record("fam_1", "missing_average", [True, True], 1.05),
            _record("fam_1", "missing_average", [True, True], 1.05),
            _record("fam_2", "rational_linear_equation", [False, False], 0.05),
            _record("fam_2", "rational_linear_equation", [False, False], 0.05),
        ],
        samples_per_prompt=2,
        reward_std_threshold=0.05,
        parse_complete_threshold=0.85,
    )

    assert summary["reward_std"] > 0.05
    assert not summary["has_prompt_level_contrast"]
    assert not summary["passes_audit_gate"]
