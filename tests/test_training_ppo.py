from reasoning_post_training.methods.ppo import ppo_rule_reward


def test_ppo_rule_reward_matches_gsm8k_rule_reward_shape():
    assert ppo_rule_reward("Reasoning. Final answer: 42", "42") == 1.1
    assert ppo_rule_reward("Reasoning. Final answer: 12", "42") == 0.1
    assert ppo_rule_reward("No extractable answer", "42") == -0.1
