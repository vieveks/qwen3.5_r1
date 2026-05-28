from iso_rlvr.train.packed_grpo_trl import (
    make_packed_trl_reward_func,
    packed_trl_reward_config,
    packed_trl_rewards,
)


def test_packed_trl_rewards_score_plain_string_completions():
    rewards = packed_trl_rewards(
        completions=[
            "<answers>\n<answer_1>2</answer_1>\n<answer_2>3</answer_2>\n</answers>",
            "<answers>\n<answer_1>2</answer_1>\n<answer_2>4</answer_2>\n</answers>",
        ],
        gold_answers=[["2", "3"], ["2", "3"]],
        config=packed_trl_reward_config({"family_bonus_enabled": False}),
    )

    assert rewards == [1.05, 0.55]


def test_packed_trl_reward_func_accepts_dataset_columns_as_kwargs():
    reward_func = make_packed_trl_reward_func({"family_bonus_enabled": False})

    rewards = reward_func(
        prompts=["prompt 1", "prompt 2"],
        completions=[
            "<answers>\n<answer_1>2</answer_1>\n<answer_2>3</answer_2>\n</answers>",
            "<answers>\n<answer_1>2</answer_1>\n<answer_2>4</answer_2>\n</answers>",
        ],
        completion_ids=[[1, 2, 3], [1, 2, 3, 4]],
        gold_answers=[["2", "3"], ["2", "3"]],
        family_id=["fam_1", "fam_2"],
        family_type=["toy", "toy"],
        variant_ids=[["a", "b"], ["c", "d"]],
        num_variants=[2, 2],
        prompt_format=["xml", "xml"],
    )

    assert len(rewards) == 2
    assert rewards[0] == 1.05
    assert rewards[1] < 0.55


def test_packed_trl_rewards_accept_chat_style_completions():
    rewards = packed_trl_rewards(
        completions=[
            [
                {
                    "role": "assistant",
                    "content": "<answers>\n<answer_1>2</answer_1>\n<answer_2>3</answer_2>\n</answers>",
                }
            ]
        ],
        gold_answers=[["2", "3"]],
        config=packed_trl_reward_config({"family_bonus_enabled": False}),
    )

    assert rewards == [1.05]


def test_packed_trl_rewards_reject_mismatched_lengths():
    try:
        packed_trl_rewards(completions=["x"], gold_answers=[])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected mismatched inputs to raise ValueError")


def test_packed_trl_rewards_reject_mismatched_completion_ids():
    try:
        packed_trl_rewards(completions=["x"], gold_answers=[["1"]], completion_ids=[])
    except ValueError as exc:
        assert "completion_ids" in str(exc)
    else:
        raise AssertionError("Expected mismatched completion_ids to raise ValueError")
