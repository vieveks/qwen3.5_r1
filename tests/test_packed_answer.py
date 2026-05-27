import pytest

from iso_rlvr.rewards.packed_answer import parse_packed_answers


def test_parse_canonical_answer_lines():
    parsed = parse_packed_answers(
        """
        Answer 1: 3
        Answer 2: 7
        Answer 3: 21
        Answer 4: 4
        """,
        expected_count=4,
    )

    assert parsed.answers == ["3", "7", "21", "4"]
    assert parsed.missing_indices == []
    assert parsed.extra_answers == []
    assert parsed.mode == "indexed"
    assert parsed.complete


def test_parse_xml_answer_block():
    parsed = parse_packed_answers(
        """
        <answers>
        <answer_1>3</answer_1>
        <answer_2>7</answer_2>
        </answers>
        """,
        expected_count=2,
    )

    assert parsed.answers == ["3", "7"]
    assert parsed.missing_indices == []
    assert parsed.extra_answers == []
    assert parsed.mode == "xml"
    assert parsed.complete


def test_parse_xml_fraction_answers():
    parsed = parse_packed_answers(
        r"""
        <answers>
        <answer_1>16/5</answer_1>
        <answer_2>\frac{-3}{7}</answer_2>
        </answers>
        """,
        expected_count=2,
    )

    assert parsed.answers == ["16/5", "-3/7"]
    assert parsed.complete


def test_parse_xml_marks_missing_and_extra_answers():
    parsed = parse_packed_answers(
        """
        <answers>
        <answer_1>3</answer_1>
        <answer_3>21</answer_3>
        <answer_5>99</answer_5>
        </answers>
        """,
        expected_count=4,
    )

    assert parsed.answers == ["3", None, "21", None]
    assert parsed.missing_indices == [2, 4]
    assert parsed.extra_answers == ["99"]
    assert not parsed.complete


def test_parse_xml_duplicate_answer_as_extra():
    parsed = parse_packed_answers(
        """
        <answers>
        <answer_1>3</answer_1>
        <answer_1>4</answer_1>
        <answer_2>5</answer_2>
        </answers>
        """,
        expected_count=2,
    )

    assert parsed.answers == ["3", "5"]
    assert parsed.extra_answers == ["4"]
    assert not parsed.complete


def test_parse_xml_rejects_prose_inside_answer_tag():
    parsed = parse_packed_answers(
        """
        <answers>
        <answer_1>the answer is 3</answer_1>
        <answer_2>7</answer_2>
        </answers>
        """,
        expected_count=2,
    )

    assert parsed.answers == [None, "7"]
    assert parsed.missing_indices == [1]
    assert not parsed.complete


def test_parse_xml_attempt_does_not_fall_back_to_boxed_or_indexed_answers():
    parsed = parse_packed_answers(
        r"""
        <answers>
        <answer_1>the answer is \boxed{3}</answer_1>
        </answers>

        Answer 1: 3
        Answer 2: 7
        """,
        expected_count=2,
    )

    assert parsed.answers == [None, None]
    assert parsed.mode == "xml"
    assert not parsed.complete


def test_parse_marks_missing_and_extra_indexed_answers():
    parsed = parse_packed_answers(
        """
        Answer 1: 3
        Answer 3: 21
        Answer 5: 99
        """,
        expected_count=4,
    )

    assert parsed.answers == ["3", None, "21", None]
    assert parsed.missing_indices == [2, 4]
    assert parsed.extra_answers == ["99"]
    assert not parsed.complete


def test_parse_duplicate_index_as_extra_answer():
    parsed = parse_packed_answers(
        """
        Answer 1: 3
        Answer 1: 4
        Answer 2: 5
        """,
        expected_count=2,
    )

    assert parsed.answers == ["3", "5"]
    assert parsed.extra_answers == ["4"]
    assert not parsed.complete


def test_parse_fallback_numbered_lines():
    parsed = parse_packed_answers(
        """
        1) 3
        2) 7
        3) 21
        4) 4
        """,
        expected_count=4,
    )

    assert parsed.answers == ["3", "7", "21", "4"]
    assert parsed.complete


def test_parse_fallback_problem_lines():
    parsed = parse_packed_answers(
        """
        Problem 1 final answer: 3
        Problem 2 answer: 7
        """,
        expected_count=2,
    )

    assert parsed.answers == ["3", "7"]
    assert parsed.complete


def test_parse_does_not_treat_problem_statement_as_answer():
    parsed = parse_packed_answers(
        """
        Problem 1: Solve for x: -6x + 22/3 = -2/3.
        Problem 2: Solve for x: -8x + 16/5 = -112/15.
        """,
        expected_count=2,
    )

    assert parsed.answers == [None, None]
    assert not parsed.complete


def test_parse_boxed_answers_in_order():
    parsed = parse_packed_answers(
        r"""
        \boxed{3}
        \boxed{7}
        \boxed{21}
        """,
        expected_count=2,
    )

    assert parsed.answers == ["3", "7"]
    assert parsed.extra_answers == ["21"]
    assert parsed.mode == "boxed"


def test_parse_latex_fraction_answer_lines():
    parsed = parse_packed_answers(
        r"""
        Answer 1: \frac{16}{5}
        Answer 2: \boxed{\frac{-3}{7}}
        """,
        expected_count=2,
    )

    assert parsed.answers == ["16/5", "-3/7"]
    assert parsed.complete


def test_parse_boxed_answers_inside_problem_sections():
    parsed = parse_packed_answers(
        r"""
        Problem 1: solve it.
        The answer is:
        \[
        \boxed{\frac{16}{5}}
        \]

        Problem 2: solve it.
        The answer is \boxed{7}.
        """,
        expected_count=2,
    )

    assert parsed.answers == ["16/5", "7"]
    assert parsed.complete


def test_parse_missing_when_no_supported_answers():
    parsed = parse_packed_answers("I am not sure.", expected_count=3)

    assert parsed.answers == [None, None, None]
    assert parsed.missing_indices == [1, 2, 3]
    assert parsed.extra_answers == []
    assert parsed.mode == "missing"


def test_parse_requires_positive_expected_count():
    with pytest.raises(ValueError, match="expected_count must be positive"):
        parse_packed_answers("Answer 1: 3", expected_count=0)
