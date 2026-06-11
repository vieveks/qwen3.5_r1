import pytest

from iso_rlvr.eval.summarize_family_consistency import summarize_family_consistency


def _exploded_row(
    family_id: str,
    variant_idx: int,
    correct: bool,
    family_type: str = "missing_average",
) -> dict:
    return {
        "family_id": family_id,
        "row_id": f"{family_id}::{family_id}_v{variant_idx}",
        "family_type": family_type,
        "num_variants": 1,
        "correctness": [correct],
        "parse_complete": True,
    }


def test_summarize_family_consistency_requires_all_variants_correct_across_contexts():
    rows = [
        _exploded_row("fam_000001", 0, True),
        _exploded_row("fam_000001", 1, True),
        _exploded_row("fam_000002", 0, True),
        _exploded_row("fam_000002", 1, False),
    ]

    summary = summarize_family_consistency(rows)

    assert summary["families"] == 2
    assert summary["variant_examples"] == 4
    assert summary["accuracy"] == 0.75
    assert summary["cross_context_family_accuracy"] == 0.5
    assert summary["parse_complete_rate"] == 1.0


def test_summarize_family_consistency_breaks_down_by_family_type():
    rows = [
        _exploded_row("fam_000001", 0, True, "missing_average"),
        _exploded_row("fam_000001", 1, True, "missing_average"),
        _exploded_row("fam_000002", 0, False, "rational_linear_equation"),
        _exploded_row("fam_000002", 1, False, "rational_linear_equation"),
    ]

    summary = summarize_family_consistency(rows)

    by_type = summary["by_family_type"]
    assert by_type["missing_average"]["cross_context_family_accuracy"] == 1.0
    assert by_type["rational_linear_equation"]["cross_context_family_accuracy"] == 0.0


def test_summarize_family_consistency_rejects_packed_rows():
    rows = [
        {
            "family_id": "fam_000001",
            "family_type": "missing_average",
            "num_variants": 2,
            "correctness": [True, True],
            "parse_complete": True,
        }
    ]

    with pytest.raises(ValueError, match="single-variant"):
        summarize_family_consistency(rows)


def test_summarize_family_consistency_handles_empty_input():
    summary = summarize_family_consistency([])

    assert summary["families"] == 0
    assert summary["cross_context_family_accuracy"] == 0.0
    assert summary["by_family_type"] == {}
