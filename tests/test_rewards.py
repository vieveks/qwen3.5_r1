from iso_rlvr.rewards.answer import extract_answer, is_correct
from iso_rlvr.rewards.iso import ScoredResponse, family_consistency, iso_reward_values, iso_rewards


def test_extract_answer_prefers_answer_marker():
    assert extract_answer("We compute 12. Answer: 9") == "9"


def test_fraction_correctness():
    assert is_correct("2/4", "1/2")


def test_family_consistency_requires_all_correct():
    scored = [
        ScoredResponse("f1", "v1", "2", "Answer: 2", True, "2", 5),
        ScoredResponse("f1", "v2", "3", "Answer: 0", False, "0", 5),
    ]
    assert family_consistency(scored)["f1"] == 0.0
    rewards = iso_rewards(scored, lambda_iso=0.5)
    assert rewards["v1"] == 1.0
    assert rewards["v2"] == 0.0
    assert iso_reward_values(scored, lambda_iso=0.5) == [1.0, 0.0]


def test_iso_reward_values_preserves_duplicate_variant_samples():
    scored = [
        ScoredResponse("f1", "v1", "2", "Answer: 2", True, "2", 5),
        ScoredResponse("f1", "v1", "2", "Answer: 0", False, "0", 5),
    ]
    assert iso_reward_values(scored, lambda_iso=0.5) == [1.0, 0.0]
