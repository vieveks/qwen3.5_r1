from iso_rlvr.eval.run_packed_eval import summarize_packed_eval_rows


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
            _row("missing_average", [True, True], True, 1.55),
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
    assert summary["by_family_type"]["missing_average"]["accuracy"] == 1.0
    assert summary["by_family_type"]["rational_linear_equation"]["accuracy"] == 0.5
    assert "by_family_type" not in summary["by_family_type"]["missing_average"]


def test_summarize_packed_eval_rows_handles_empty_input():
    summary = summarize_packed_eval_rows([])

    assert summary["examples"] == 0
    assert summary["by_family_type"] == {}
