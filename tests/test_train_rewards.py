import pytest

from iso_rlvr.rewards.iso import ScoredResponse
from iso_rlvr.train.grpo_lite import reward_values


def test_independent_reward_values_ignore_family_bonus():
    scored = [
        ScoredResponse("f1", "v1", "2", "Answer: 2", True, "2", 5),
        ScoredResponse("f1", "v2", "3", "Answer: 0", False, "0", 5),
    ]
    assert reward_values(scored, {"reward_mode": "independent", "lambda_iso": 1.0}) == [1.0, 0.0]


def test_iso_reward_values_include_family_bonus():
    scored = [
        ScoredResponse("f1", "v1", "2", "Answer: 2", True, "2", 5),
        ScoredResponse("f1", "v2", "3", "Answer: 3", True, "3", 5),
    ]
    assert reward_values(scored, {"reward_mode": "iso", "lambda_iso": 0.5}) == [1.5, 1.5]


def test_unknown_reward_mode_raises():
    with pytest.raises(ValueError, match="Unknown reward_mode"):
        reward_values([], {"reward_mode": "missing"})
