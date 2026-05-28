import torch

from iso_rlvr.train.packed_grpo_lite import grouped_advantages, packed_reward_config
from iso_rlvr.rewards.packed_iso import score_packed_completion


def test_grouped_advantages_are_normalized_within_prompt_group():
    rewards = torch.tensor([0.0, 1.0, 2.0, 5.0, 5.0])
    advantages = grouped_advantages(rewards, ["a", "a", "a", "b", "b"])

    assert torch.allclose(advantages[:3].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages[:3].std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(advantages[3:], torch.tensor([0.0, 0.0]))


def test_grouped_advantages_reject_mismatched_lengths():
    try:
        grouped_advantages(torch.tensor([1.0]), [])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected mismatched rewards and group ids to raise ValueError")


def test_packed_reward_config_disables_family_bonus_by_default():
    cfg = packed_reward_config({})

    scored = score_packed_completion(
        "<answers>\n<answer_1>2</answer_1>\n<answer_2>3</answer_2>\n</answers>",
        ["2", "3"],
        config=cfg,
    )

    assert scored.family_component == 0.0
    assert scored.reward == 1.05


def test_packed_reward_config_can_enable_family_bonus():
    cfg = packed_reward_config({"family_bonus_enabled": True})

    scored = score_packed_completion(
        "<answers>\n<answer_1>2</answer_1>\n<answer_2>3</answer_2>\n</answers>",
        ["2", "3"],
        config=cfg,
    )

    assert scored.family_component == 0.5
    assert scored.reward == 1.55
