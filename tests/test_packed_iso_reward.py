import pytest

from iso_rlvr.rewards.packed_iso import (
    PackedRewardConfig,
    packed_reward_values,
    score_packed_completion,
)


def test_all_correct_gets_correctness_format_and_full_family_bonus():
    scored = score_packed_completion(
        """
        Answer 1: 3
        Answer 2: 7
        Answer 3: 21
        Answer 4: 4
        """,
        ["3", "7", "21", "4"],
    )

    assert scored.correctness == [True, True, True, True]
    assert scored.family_mean == 1.0
    assert scored.all_family_correct
    assert scored.format_component == pytest.approx(0.05)
    assert scored.family_component == pytest.approx(0.50)
    assert scored.reward == pytest.approx(1.55)


def test_partial_family_bonus_is_correctness_gated():
    scored = score_packed_completion(
        """
        Answer 1: 3
        Answer 2: 0
        Answer 3: 21
        Answer 4: 0
        """,
        ["3", "7", "21", "4"],
    )

    assert scored.correctness == [True, False, True, False]
    assert scored.family_mean == 0.5
    assert not scored.all_family_correct
    assert scored.family_component == pytest.approx(0.0625)
    assert scored.reward == pytest.approx(0.6125)


def test_wrong_answers_do_not_receive_family_bonus_from_correct_siblings():
    scored = score_packed_completion(
        """
        Answer 1: 3
        Answer 2: 0
        """,
        ["3", "7"],
    )

    assert scored.correctness == [True, False]
    assert scored.family_component == pytest.approx(0.0625)
    assert scored.reward == pytest.approx(0.6125)


def test_missing_answers_receive_format_penalty():
    scored = score_packed_completion("Answer 1: 3", ["3", "7"])

    assert scored.parse.answers == ["3", None]
    assert scored.parse.missing_indices == [2]
    assert scored.format_component == pytest.approx(-0.025)
    assert scored.reward == pytest.approx(0.5375)


def test_extra_answers_are_penalized():
    scored = score_packed_completion(
        """
        Answer 1: 3
        Answer 2: 7
        Answer 3: 99
        """,
        ["3", "7"],
    )

    assert scored.parse.extra_answers == ["99"]
    assert scored.extra_answer_penalty == pytest.approx(0.05)
    assert scored.reward == pytest.approx(1.50)


def test_wrong_length_penalty_applies_only_to_wrong_fraction():
    config = PackedRewardConfig(length_penalty_weight=0.05, token_cap=100)
    scored = score_packed_completion(
        """
        Answer 1: 3
        Answer 2: 0
        """,
        ["3", "7"],
        response_tokens=80,
        config=config,
    )

    assert scored.length_penalty == pytest.approx(0.02)
    assert scored.reward == pytest.approx(0.5925)


def test_wrong_length_penalty_does_not_apply_when_all_correct():
    config = PackedRewardConfig(length_penalty_weight=0.05, token_cap=100)
    scored = score_packed_completion(
        """
        Answer 1: 3
        Answer 2: 7
        """,
        ["3", "7"],
        response_tokens=100,
        config=config,
    )

    assert scored.length_penalty == 0.0
    assert scored.reward == pytest.approx(1.55)


def test_packed_reward_values_matches_trl_list_style_inputs():
    rewards = packed_reward_values(
        completions=[
            "Answer 1: 3\nAnswer 2: 7",
            "Answer 1: 3\nAnswer 2: 0",
        ],
        gold_answers=[["3", "7"], ["3", "7"]],
    )

    assert rewards == pytest.approx([1.55, 0.6125])


def test_packed_reward_values_validates_batch_lengths():
    with pytest.raises(ValueError, match="same length"):
        packed_reward_values(["Answer 1: 3"], [])

    with pytest.raises(ValueError, match="response_tokens"):
        packed_reward_values(["Answer 1: 3"], [["3"]], response_tokens=[1, 2])


def test_score_requires_gold_answers():
    with pytest.raises(ValueError, match="gold_answers must not be empty"):
        score_packed_completion("Answer 1: 3", [])


def test_score_does_not_credit_answers_inside_think():
    scored = score_packed_completion(
        """
        <think>
        Answer 1: 3
        Answer 2: 7
        </think>
        """,
        ["3", "7"],
    )

    assert scored.parse.answers == [None, None]
    assert scored.correctness == [False, False]
    assert scored.reward < 0
