from iso_rlvr.data.build_packed_dataset import (
    pack_dataset,
    pack_family,
    parse_family_types,
)


def _row(family_id: str, variant_idx: int, family_type: str = "missing_average") -> dict:
    answer = str(variant_idx + 10)
    return {
        "family_id": family_id,
        "variant_id": f"{family_id}_v{variant_idx}",
        "family_type": family_type,
        "problem": f"Problem text {variant_idx}?",
        "answer": answer,
        "metadata": {"idx": variant_idx},
    }


def test_pack_family_creates_stateless_trl_columns():
    rows = [_row("fam_000001", 1), _row("fam_000001", 0)]

    packed = pack_family(rows)

    assert packed["family_id"] == "fam_000001"
    assert packed["family_type"] == "missing_average"
    assert packed["num_variants"] == 2
    assert packed["variant_ids"] == ["fam_000001_v0", "fam_000001_v1"]
    assert packed["problems"] == ["Problem text 0?", "Problem text 1?"]
    assert packed["gold_answers"] == ["10", "11"]
    assert packed["metadata"] == [{"idx": 0}, {"idx": 1}]
    assert "Return only the final answers" in packed["prompt"]
    assert "Problem 1: Problem text 0?" in packed["prompt"]
    assert "Problem 2: Problem text 1?" in packed["prompt"]
    assert "Answer 1: <number>" in packed["prompt"]
    assert "Answer 2: <number>" in packed["prompt"]


def test_pack_dataset_filters_curriculum_family_types_and_variant_count():
    rows = [
        _row("fam_000001", 0, "missing_average"),
        _row("fam_000001", 1, "missing_average"),
        _row("fam_000002", 0, "chinese_remainder"),
        _row("fam_000002", 1, "chinese_remainder"),
        _row("fam_000003", 0, "rational_linear_equation"),
    ]

    packed = pack_dataset(
        rows,
        include_family_types={"missing_average", "rational_linear_equation"},
        expected_variants=2,
    )

    assert [row["family_id"] for row in packed] == ["fam_000001"]


def test_pack_dataset_can_limit_variants_per_family_for_smoke_tests():
    rows = [
        _row("fam_000001", 0),
        _row("fam_000001", 1),
        _row("fam_000001", 2),
        _row("fam_000001", 3),
    ]

    packed = pack_dataset(rows, expected_variants=4, max_variants_per_family=2)

    assert len(packed) == 1
    assert packed[0]["num_variants"] == 2
    assert packed[0]["variant_ids"] == ["fam_000001_v0", "fam_000001_v1"]
    assert "Answer 2: <number>" in packed[0]["prompt"]
    assert "Answer 3: <number>" not in packed[0]["prompt"]


def test_parse_family_types_supports_repeated_and_comma_separated_values():
    assert parse_family_types(["missing_average,rational_linear_equation"]) == {
        "missing_average",
        "rational_linear_equation",
    }
    assert parse_family_types(["missing_average", "rational_linear_equation"]) == {
        "missing_average",
        "rational_linear_equation",
    }
    assert parse_family_types(None) is None
