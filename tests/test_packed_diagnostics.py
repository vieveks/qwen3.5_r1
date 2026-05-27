import pytest

from iso_rlvr.eval.packed_diagnostics import diagnose_packed_completion, diagnose_packed_parse
from iso_rlvr.rewards.packed_answer import PackedAnswerParse


def test_diagnose_repeated_and_copied_answers():
    diagnostics = diagnose_packed_completion(
        """
        Answer 1: 5
        Answer 2: 5
        Answer 3: 5
        """,
        ["1", "2", "3"],
    )

    assert diagnostics.repeated_answer
    assert diagnostics.copied_answer_indices == [2, 3]
    assert diagnostics.suspicious


def test_diagnose_only_first_answer_and_missing_later_answers():
    diagnostics = diagnose_packed_completion("Answer 1: 5", ["5", "6", "7"])

    assert diagnostics.only_first_answer
    assert diagnostics.missing_indices == [2, 3]
    assert diagnostics.answer_count_mismatch
    assert diagnostics.suspicious


def test_diagnose_extra_answers_as_count_mismatch():
    diagnostics = diagnose_packed_completion(
        """
        Answer 1: 5
        Answer 2: 6
        Answer 3: 7
        """,
        ["5", "6"],
    )

    assert diagnostics.extra_answer_count == 1
    assert diagnostics.answer_count_mismatch


def test_diagnose_same_wrong_additive_offset():
    diagnostics = diagnose_packed_completion(
        """
        Answer 1: 11
        Answer 2: 12
        Answer 3: 13
        """,
        ["10", "11", "12"],
    )

    assert diagnostics.same_wrong_additive_offset
    assert not diagnostics.same_wrong_multiplicative_offset
    assert diagnostics.suspicious


def test_diagnose_same_wrong_multiplicative_offset():
    diagnostics = diagnose_packed_completion(
        """
        Answer 1: 20
        Answer 2: 40
        Answer 3: 60
        """,
        ["10", "20", "30"],
    )

    assert diagnostics.same_wrong_multiplicative_offset
    assert not diagnostics.same_wrong_additive_offset
    assert diagnostics.suspicious


def test_clean_completion_is_not_suspicious():
    diagnostics = diagnose_packed_completion(
        """
        Answer 1: 10
        Answer 2: 20
        Answer 3: 30
        """,
        ["10", "20", "30"],
    )

    assert not diagnostics.repeated_answer
    assert diagnostics.copied_answer_indices == []
    assert not diagnostics.answer_count_mismatch
    assert not diagnostics.suspicious


def test_diagnose_validates_parse_length():
    parsed = PackedAnswerParse(
        answers=["1"],
        missing_indices=[],
        extra_answers=[],
        mode="indexed",
    )

    with pytest.raises(ValueError, match="same length"):
        diagnose_packed_parse(parsed, ["1", "2"])
