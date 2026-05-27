from iso_rlvr.data.build_format_sft_dataset import (
    build_format_sft_dataset,
    build_sft_row,
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
    assert row["prompt_format"] == "xml"
    assert row["completion"] == (
        "<answers>\n"
        "<answer_1>16/5</answer_1>\n"
        "<answer_2>7</answer_2>\n"
        "</answers>"
    )
    assert row["text"] == f"{row['prompt']}\n{row['completion']}"


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
