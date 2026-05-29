from iso_rlvr.data.build_format_sft_dataset import (
    build_chinese_remainder_trace,
    build_missing_average_trace,
    build_rational_linear_trace,
    build_rational_system_trace,
    build_format_sft_dataset,
    build_sft_row,
    build_sft_rows,
    build_xml_completion,
    split_by_family_id,
)
from iso_rlvr.io import read_jsonl, write_jsonl


def _packed_row(family_id: str, answer: str = "3") -> dict:
    return {
        "family_id": family_id,
        "family_type": "rational_linear_equation",
        "variant_ids": [f"{family_id}_v0", f"{family_id}_v1"],
        "num_variants": 2,
        "gold_answers": [answer, "7"],
        "prompt_format": "xml",
        "prompt": "Problem 1: ...\n\nProblem 2: ...",
    }


def _missing_average_row(family_id: str = "fam_000001") -> dict:
    return {
        "family_id": family_id,
        "family_type": "missing_average",
        "variant_ids": [f"{family_id}_v0", f"{family_id}_v1"],
        "num_variants": 2,
        "gold_answers": ["32", "32"],
        "metadata": [
            {"known": "56,29,13,15", "final_average": "29", "missing": 32},
            {"known": "30,62,10,21,75,60,64", "final_average": "177/4", "missing": 32},
        ],
        "prompt_format": "xml",
        "prompt": "Problem 1: ...\n\nProblem 2: ...",
    }


def _rational_linear_row(family_id: str = "fam_000020") -> dict:
    return {
        "family_id": family_id,
        "family_type": "rational_linear_equation",
        "variant_ids": [f"{family_id}_v0", f"{family_id}_v1"],
        "num_variants": 2,
        "gold_answers": ["8/5", "8/5"],
        "metadata": [
            {"a": -5, "b": "-13/4", "c": "-45/4", "solution": "8/5"},
            {"a": 4, "b": "1/3", "c": "101/15", "solution": "8/5"},
        ],
        "prompt_format": "xml",
        "prompt": "Problem 1: ...\n\nProblem 2: ...",
    }


def _rational_system_row(family_id: str = "fam_000030") -> dict:
    return {
        "family_id": family_id,
        "family_type": "rational_system_target",
        "variant_ids": [f"{family_id}_v0", f"{family_id}_v1"],
        "num_variants": 2,
        "gold_answers": ["-1/2", "-1/2"],
        "metadata": [
            {
                "x": "1/2",
                "y": "-1",
                "target_name": "x + y",
                "target": "-1/2",
                "a": 2,
                "b": 3,
                "c": 1,
                "d": -1,
            },
            {
                "x": "1/2",
                "y": "-1",
                "target_name": "x + y",
                "target": "-1/2",
                "a": -3,
                "b": 2,
                "c": 5,
                "d": 1,
            },
        ],
        "prompt_format": "xml",
        "prompt": "Problem 1: ...\n\nProblem 2: ...",
    }


def _chinese_remainder_row(family_id: str = "fam_000040") -> dict:
    return {
        "family_id": family_id,
        "family_type": "chinese_remainder",
        "variant_ids": [f"{family_id}_v0", f"{family_id}_v1"],
        "num_variants": 2,
        "gold_answers": ["58", "58"],
        "metadata": [
            {"answer": 58, "mod_a": 7, "mod_b": 9},
            {"answer": 58, "mod_a": 11, "mod_b": 13},
        ],
        "prompt_format": "xml",
        "prompt": "Problem 1: ...\n\nProblem 2: ...",
    }


def test_build_xml_completion_uses_answer_only_xml_block():
    completion = build_xml_completion(["16/5", "-3"])

    assert completion == (
        "<answers>\n"
        "<answer_1>16/5</answer_1>\n"
        "<answer_2>-3</answer_2>\n"
        "</answers>"
    )


def test_build_sft_row_preserves_reward_context_and_adds_text_field():
    row = build_sft_row(_packed_row("fam_000001", answer="16/5"))

    assert row["family_id"] == "fam_000001"
    assert row["family_type"] == "rational_linear_equation"
    assert row["variant_ids"] == ["fam_000001_v0", "fam_000001_v1"]
    assert row["num_variants"] == 2
    assert row["gold_answers"] == ["16/5", "7"]
    assert row["target_style"] == "answer_only_xml"
    assert row["prompt_format"] == "xml"
    assert row["completion"] == (
        "<answers>\n"
        "<answer_1>16/5</answer_1>\n"
        "<answer_2>7</answer_2>\n"
        "</answers>"
    )
    assert row["text"] == f"{row['prompt']}\n{row['completion']}"


def test_build_missing_average_trace_uses_prompt_metadata_values():
    trace = build_missing_average_trace(_missing_average_row())

    assert trace == (
        "Problem 1 sum needed: 29 x 5 = 145\n"
        "Problem 1 known sum: 56 + 29 + 13 + 15 = 113\n"
        "Problem 1 missing value: 145 - 113 = 32\n\n"
        "Problem 2 sum needed: 177/4 x 8 = 354\n"
        "Problem 2 known sum: 30 + 62 + 10 + 21 + 75 + 60 + 64 = 322\n"
        "Problem 2 missing value: 354 - 322 = 32"
    )


def test_build_sft_row_can_add_missing_average_trace_before_xml():
    row = build_sft_row(_missing_average_row(), missing_average_traces=True)

    assert row["target_style"] == "missing_average_trace_xml"
    assert row["completion"].startswith("Problem 1 sum needed: 29 x 5 = 145")
    assert row["completion"].endswith(
        "<answers>\n"
        "<answer_1>32</answer_1>\n"
        "<answer_2>32</answer_2>\n"
        "</answers>"
    )


def test_build_rational_linear_trace_uses_prompt_metadata_values():
    trace = build_rational_linear_trace(_rational_linear_row())

    assert trace == (
        "Problem 1 isolate: -5x = -45/4 - (-13/4) = -8\n"
        "Problem 1 divide: x = -8 / -5 = 8/5\n\n"
        "Problem 2 isolate: 4x = 101/15 - (1/3) = 32/5\n"
        "Problem 2 divide: x = 32/5 / 4 = 8/5"
    )


def test_build_sft_row_can_add_rational_linear_trace_before_xml():
    row = build_sft_row(_rational_linear_row(), rational_linear_traces=True)

    assert row["target_style"] == "rational_linear_trace_xml"
    assert row["completion"].startswith("Problem 1 isolate: -5x = -45/4 - (-13/4) = -8")
    assert row["completion"].endswith(
        "<answers>\n"
        "<answer_1>8/5</answer_1>\n"
        "<answer_2>8/5</answer_2>\n"
        "</answers>"
    )


def test_build_rational_system_trace_uses_fixed_elimination_method():
    trace = build_rational_system_trace(_rational_system_row())

    assert trace == (
        "Problem 1 equations: 2x + (3)y = -2; 1x + (-1)y = 3/2\n"
        "Problem 1 eliminate y: -1 times first minus 3 times second gives -5x = -5/2\n"
        "Problem 1 solve x: x = -5/2 / -5 = 1/2\n"
        "Problem 1 eliminate x: 2 times second minus 1 times first gives -5y = 5\n"
        "Problem 1 solve y: y = 5 / -5 = -1\n"
        "Problem 1 target: x + y = (1/2) + (-1) = -1/2\n\n"
        "Problem 2 equations: -3x + (2)y = -7/2; 5x + (1)y = 3/2\n"
        "Problem 2 eliminate y: 1 times first minus 2 times second gives -13x = -13/2\n"
        "Problem 2 solve x: x = -13/2 / -13 = 1/2\n"
        "Problem 2 eliminate x: -3 times second minus 5 times first gives -13y = 13\n"
        "Problem 2 solve y: y = 13 / -13 = -1\n"
        "Problem 2 target: x + y = (1/2) + (-1) = -1/2"
    )


def test_build_sft_row_can_add_rational_system_trace_before_xml():
    row = build_sft_row(_rational_system_row(), rational_system_traces=True)

    assert row["target_style"] == "rational_system_trace_xml"
    assert row["completion"].startswith("Problem 1 equations: 2x + (3)y = -2")
    assert row["completion"].endswith(
        "<answers>\n"
        "<answer_1>-1/2</answer_1>\n"
        "<answer_2>-1/2</answer_2>\n"
        "</answers>"
    )


def test_build_chinese_remainder_trace_uses_numeric_solution_and_assertions():
    trace = build_chinese_remainder_trace(_chinese_remainder_row())

    assert trace == (
        "Problem 1 range: 0 <= x < 7 x 9 = 63\n"
        "Problem 1 check first congruence: 58 mod 7 = 2\n"
        "Problem 1 check second congruence: 58 mod 9 = 4\n"
        "Problem 1 least value: x = 58\n\n"
        "Problem 2 range: 0 <= x < 11 x 13 = 143\n"
        "Problem 2 check first congruence: 58 mod 11 = 3\n"
        "Problem 2 check second congruence: 58 mod 13 = 6\n"
        "Problem 2 least value: x = 58"
    )


def test_build_chinese_remainder_trace_rejects_non_coprime_moduli():
    row = _chinese_remainder_row()
    row["metadata"][0]["mod_a"] = 6
    row["metadata"][0]["mod_b"] = 9

    try:
        build_chinese_remainder_trace(row)
    except ValueError as exc:
        assert "coprime" in str(exc)
    else:
        raise AssertionError("Expected non-coprime CRT metadata to raise ValueError")


def test_build_sft_row_can_add_chinese_remainder_trace_before_xml():
    row = build_sft_row(_chinese_remainder_row(), chinese_remainder_traces=True)

    assert row["target_style"] == "chinese_remainder_trace_xml"
    assert row["completion"].startswith("Problem 1 range: 0 <= x < 7 x 9 = 63")
    assert row["completion"].endswith(
        "<answers>\n"
        "<answer_1>58</answer_1>\n"
        "<answer_2>58</answer_2>\n"
        "</answers>"
    )


def test_build_sft_row_keeps_algebra_answer_only_when_missing_trace_mode_is_enabled():
    row = build_sft_row(_packed_row("fam_000002"), missing_average_traces=True)

    assert row["target_style"] == "answer_only_xml"
    assert row["completion"] == (
        "<answers>\n"
        "<answer_1>3</answer_1>\n"
        "<answer_2>7</answer_2>\n"
        "</answers>"
    )


def test_build_sft_rows_can_include_answer_only_copy_for_missing_average():
    rows = build_sft_rows(
        _missing_average_row(),
        missing_average_traces=True,
        include_answer_only_copy=True,
    )

    assert [row["target_style"] for row in rows] == [
        "answer_only_xml",
        "missing_average_trace_xml",
    ]
    assert rows[0]["completion"].startswith("<answers>")
    assert rows[1]["completion"].startswith("Problem 1 sum needed:")


def test_split_by_family_id_keeps_families_disjoint():
    rows = [_packed_row(f"fam_{idx:06d}") for idx in range(10)]

    train_rows, heldout_rows = split_by_family_id(rows, heldout_fraction=0.2, seed=0)

    train_ids = {row["family_id"] for row in train_rows}
    heldout_ids = {row["family_id"] for row in heldout_rows}
    assert len(train_rows) == 8
    assert len(heldout_rows) == 2
    assert not train_ids & heldout_ids


def test_split_by_family_id_rejects_invalid_heldout_fraction():
    try:
        split_by_family_id([], heldout_fraction=1.0, seed=0)
    except ValueError as exc:
        assert "heldout_fraction" in str(exc)
    else:
        raise AssertionError("Expected invalid heldout_fraction to raise ValueError")


def test_build_format_sft_dataset_can_write_matching_packed_splits(tmp_path):
    input_path = tmp_path / "packed.jsonl"
    train_out = tmp_path / "sft_train.jsonl"
    heldout_out = tmp_path / "sft_heldout.jsonl"
    packed_train_out = tmp_path / "packed_train.jsonl"
    packed_heldout_out = tmp_path / "packed_heldout.jsonl"
    rows = [_packed_row(f"fam_{idx:06d}") for idx in range(10)]
    write_jsonl(input_path, rows)

    build_format_sft_dataset(
        input_path,
        train_out,
        heldout_out=heldout_out,
        packed_train_out=packed_train_out,
        packed_heldout_out=packed_heldout_out,
        heldout_fraction=0.2,
        seed=0,
    )

    sft_train_ids = {row["family_id"] for row in read_jsonl(train_out)}
    sft_heldout_ids = {row["family_id"] for row in read_jsonl(heldout_out)}
    packed_train_ids = {row["family_id"] for row in read_jsonl(packed_train_out)}
    packed_heldout_ids = {row["family_id"] for row in read_jsonl(packed_heldout_out)}
    assert sft_train_ids == packed_train_ids
    assert sft_heldout_ids == packed_heldout_ids
    assert not sft_train_ids & sft_heldout_ids


def test_build_format_sft_dataset_can_duplicate_traced_rows(tmp_path):
    input_path = tmp_path / "packed.jsonl"
    train_out = tmp_path / "sft_train.jsonl"
    rows = [_missing_average_row("fam_000001"), _packed_row("fam_000002")]
    write_jsonl(input_path, rows)

    build_format_sft_dataset(
        input_path,
        train_out,
        heldout_fraction=0.0,
        missing_average_traces=True,
        include_answer_only_copy=True,
    )

    sft_rows = read_jsonl(train_out)
    assert [row["target_style"] for row in sft_rows] == [
        "answer_only_xml",
        "missing_average_trace_xml",
        "answer_only_xml",
    ]


def test_build_format_sft_dataset_can_trace_both_target_families(tmp_path):
    input_path = tmp_path / "packed.jsonl"
    train_out = tmp_path / "sft_train.jsonl"
    rows = [_missing_average_row("fam_000001"), _rational_linear_row("fam_000020")]
    write_jsonl(input_path, rows)

    build_format_sft_dataset(
        input_path,
        train_out,
        heldout_fraction=0.0,
        missing_average_traces=True,
        rational_linear_traces=True,
    )

    sft_rows = read_jsonl(train_out)
    assert [row["target_style"] for row in sft_rows] == [
        "missing_average_trace_xml",
        "rational_linear_trace_xml",
    ]


def test_build_format_sft_dataset_can_trace_all_broad_families(tmp_path):
    input_path = tmp_path / "packed.jsonl"
    train_out = tmp_path / "sft_train.jsonl"
    rows = [
        _missing_average_row("fam_000001"),
        _rational_linear_row("fam_000020"),
        _rational_system_row("fam_000030"),
        _chinese_remainder_row("fam_000040"),
    ]
    write_jsonl(input_path, rows)

    build_format_sft_dataset(
        input_path,
        train_out,
        heldout_fraction=0.0,
        missing_average_traces=True,
        rational_linear_traces=True,
        rational_system_traces=True,
        chinese_remainder_traces=True,
    )

    sft_rows = read_jsonl(train_out)
    assert [row["target_style"] for row in sft_rows] == [
        "missing_average_trace_xml",
        "rational_linear_trace_xml",
        "rational_system_trace_xml",
        "chinese_remainder_trace_xml",
    ]
