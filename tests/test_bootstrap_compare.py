import pytest

from iso_rlvr.eval.bootstrap_compare import (
    family_metrics,
    join_by_family,
    paired_bootstrap,
    paired_sign_flip_test,
    point_estimates,
)


def _eval_row(family_id: str, correctness: list[bool], family_type: str = "missing_average") -> dict:
    return {
        "family_id": family_id,
        "family_type": family_type,
        "gold_answers": [str(idx) for idx in range(len(correctness))],
        "correctness": correctness,
    }


def test_family_metrics_counts_variants_and_all_correct():
    metrics = family_metrics(_eval_row("fam_000001", [True, False]))

    assert metrics["variant_correct"] == 1.0
    assert metrics["variant_total"] == 2.0
    assert metrics["all_family_correct"] == 0.0
    assert family_metrics(_eval_row("fam_000001", [True, True]))["all_family_correct"] == 1.0


def test_join_by_family_pairs_shared_families_and_checks_gold_answers():
    rows_a = [_eval_row("fam_000001", [True, True]), _eval_row("fam_000002", [False, True])]
    rows_b = [_eval_row("fam_000002", [True, True]), _eval_row("fam_000001", [False, False])]

    pairs = join_by_family(rows_a, rows_b)

    assert [pair["family_id"] for pair in pairs] == ["fam_000001", "fam_000002"]
    assert pairs[0]["a"]["all_family_correct"] == 1.0
    assert pairs[0]["b"]["all_family_correct"] == 0.0


def test_join_by_family_rejects_mismatched_gold_answers():
    rows_a = [_eval_row("fam_000001", [True, True])]
    rows_b = [_eval_row("fam_000001", [True, True])]
    rows_b[0]["gold_answers"] = ["99", "99"]

    with pytest.raises(ValueError, match="Gold answers differ"):
        join_by_family(rows_a, rows_b)


def test_point_estimates_match_manual_computation():
    pairs = join_by_family(
        [_eval_row("fam_000001", [True, True]), _eval_row("fam_000002", [True, False])],
        [_eval_row("fam_000001", [False, True]), _eval_row("fam_000002", [False, False])],
    )

    estimates_a = point_estimates(pairs, "a")
    estimates_b = point_estimates(pairs, "b")

    assert estimates_a["accuracy"] == 0.75
    assert estimates_a["family_accuracy"] == 0.5
    assert estimates_b["accuracy"] == 0.25
    assert estimates_b["family_accuracy"] == 0.0


def test_paired_bootstrap_is_deterministic_and_brackets_constant_delta():
    pairs = join_by_family(
        [_eval_row(f"fam_{idx:06d}", [True, True]) for idx in range(20)],
        [_eval_row(f"fam_{idx:06d}", [True, False]) for idx in range(20)],
    )

    first = paired_bootstrap(pairs, iterations=200, seed=7)
    second = paired_bootstrap(pairs, iterations=200, seed=7)

    assert first == second
    # Every family has the identical delta, so the CI collapses onto it.
    assert first["accuracy_delta"]["ci_lower"] == pytest.approx(0.5)
    assert first["accuracy_delta"]["ci_upper"] == pytest.approx(0.5)
    assert first["family_accuracy_delta"]["ci_lower"] == pytest.approx(1.0)


def test_sign_flip_test_detects_consistent_difference_and_not_no_difference():
    consistent = join_by_family(
        [_eval_row(f"fam_{idx:06d}", [True, True]) for idx in range(12)],
        [_eval_row(f"fam_{idx:06d}", [False, False]) for idx in range(12)],
    )
    no_difference = join_by_family(
        [_eval_row(f"fam_{idx:06d}", [True, False]) for idx in range(12)],
        [_eval_row(f"fam_{idx:06d}", [True, False]) for idx in range(12)],
    )

    strong = paired_sign_flip_test(consistent, "family_accuracy", iterations=2000, seed=3)
    null = paired_sign_flip_test(no_difference, "family_accuracy", iterations=2000, seed=3)

    assert strong["observed_mean_delta"] == 1.0
    assert strong["p_value"] < 0.01
    assert null["observed_mean_delta"] == 0.0
    assert null["p_value"] == 1.0
