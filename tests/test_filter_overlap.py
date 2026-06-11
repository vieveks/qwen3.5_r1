from iso_rlvr.data.filter_overlap import drop_overlapping_families


def _row(family_id: str, variant_idx: int, problem: str) -> dict:
    return {
        "family_id": family_id,
        "variant_id": f"{family_id}_v{variant_idx}",
        "family_type": "missing_average",
        "problem": problem,
        "answer": "7",
    }


def test_drop_overlapping_families_drops_whole_family_on_any_collision():
    train_rows = [_row("train_fam", 0, "shared problem text")]
    rows = [
        _row("fam_000001", 0, "shared problem text"),
        _row("fam_000001", 1, "unique problem a"),
        _row("fam_000002", 0, "unique problem b"),
        _row("fam_000002", 1, "unique problem c"),
    ]

    kept, report = drop_overlapping_families(rows, train_rows)

    assert [row["family_id"] for row in kept] == ["fam_000002", "fam_000002"]
    assert report["dropped_families"] == 1
    assert report["dropped_family_ids"] == ["fam_000001"]
    assert report["kept_rows"] == 2


def test_drop_overlapping_families_keeps_everything_when_disjoint():
    train_rows = [_row("train_fam", 0, "train-only problem")]
    rows = [_row("fam_000001", 0, "eval-only problem")]

    kept, report = drop_overlapping_families(rows, train_rows)

    assert len(kept) == 1
    assert report["dropped_families"] == 0
