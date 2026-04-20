from grpo_reasoning.training.grpo import gsm8k_grpo_reward_func


def test_gsm8k_grpo_reward_func_rewards_correct_final_answer():
    rewards = gsm8k_grpo_reward_func(
        prompts=["p1", "p2"],
        completions=["Reasoning. Final answer: 4", "No answer here"],
        gold_answer=["4", "5"],
    )

    assert rewards[0] == 1.1
    assert rewards[1] == -0.1

