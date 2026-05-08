from reasoning_post_training.methods.ppo import RuleRewardModel, ppo_rule_reward


def test_ppo_rule_reward_matches_gsm8k_rule_reward_shape():
    assert ppo_rule_reward("Reasoning. Final answer: 42", "42") == 1.1
    assert ppo_rule_reward("Reasoning. Final answer: 12", "42") == 0.1
    assert ppo_rule_reward("No extractable answer", "42") == -0.1


def test_rule_reward_model_splits_chat_template_text():
    model = RuleRewardModel(tokenizer=None, gold_by_question={})

    question, completion = model._extract_question_and_completion(
        "system\nSolve.\nuser\nProblem:\nWhat is 2+2?\nassistant\nFinal answer: 4"
    )

    assert question == "What is 2+2?"
    assert completion == "Final answer: 4"
