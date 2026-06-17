from iso_rlvr.data.filter_by_pass_rate import (
    compute_pass_rates,
    filter_by_pass_rate,
)


def _audit(family_id: str, family_type: str, passes: list[bool]) -> list[dict]:
    return [
        {
            "family_id": family_id,
            "family_type": family_type,
            "all_family_correct": p,
        }
        for p in passes
    ]


def _dataset_row(family_id: str, family_type: str) -> dict:
    return {"family_id": family_id, "family_type": family_type, "prompt": "p"}


def test_compute_pass_rates_aggregates_per_family():
    records = _audit("fam_a", "modular", [True, False, True, False])
    stats = compute_pass_rates(records)
    assert stats["fam_a"]["pass_rate"] == 0.5
    assert stats["fam_a"]["pass_count"] == 2
    assert stats["fam_a"]["samples"] == 4


def test_filter_keeps_only_contrastful_families():
    audit = (
        _audit("all_wrong", "system", [False, False, False, False])
        + _audit("all_right", "proportional", [True, True, True, True])
        + _audit("contrast", "modular", [True, False, False, False])
    )
    dataset = [
        _dataset_row("all_wrong", "system"),
        _dataset_row("all_right", "proportional"),
        _dataset_row("contrast", "modular"),
    ]

    kept, report = filter_by_pass_rate(audit, dataset)

    assert [r["family_id"] for r in kept] == ["contrast"]
    assert report["kept_families"] == 1
    assert report["dropped_all_wrong"] == 1
    assert report["dropped_all_correct"] == 1
    assert report["by_family_type"]["modular"] == {"audited": 1, "kept": 1}


def test_filter_band_is_strict_and_configurable():
    audit = _audit("mid", "modular", [True, True, False, False])  # pass_rate 0.5
    dataset = [_dataset_row("mid", "modular")]

    # Strict default band keeps it.
    kept, _ = filter_by_pass_rate(audit, dataset)
    assert len(kept) == 1

    # Narrow band above 0.5 drops it (strict upper bound).
    kept_narrow, _ = filter_by_pass_rate(audit, dataset, pass_low=0.5, pass_high=0.9)
    assert kept_narrow == []
