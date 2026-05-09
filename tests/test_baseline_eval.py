from pathlib import Path

from reasoning_post_training.methods import baseline


def test_evaluate_gsm8k_model_scores_new_records(tmp_path: Path, monkeypatch):
    dataset = [
        {
            "question": "What is 2 + 2?",
            "prompt": "What is 2 + 2?",
            "gold_answer": "4",
            "answer": "#### 4",
        },
        {
            "question": "What is 3 + 3?",
            "prompt": "What is 3 + 3?",
            "gold_answer": "6",
            "answer": "#### 6",
        },
    ]

    monkeypatch.setattr(baseline, "load_gsm8k_split", lambda split, subset: dataset)
    monkeypatch.setattr(
        baseline,
        "load_model_and_tokenizer",
        lambda model_name_or_path, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        baseline,
        "generate_batch",
        lambda model, tokenizer, prompts, **kwargs: [
            f"We compute it. Final answer: {4 if prompt.endswith('2?') else 6}"
            for prompt in prompts
        ],
    )
    monkeypatch.setattr(baseline, "cuda_memory_summary", lambda: {})

    metrics = baseline.evaluate_gsm8k_model(
        model_name_or_path="dummy",
        adapter=None,
        split="test",
        subset="main",
        limit=2,
        start=0,
        batch_size=2,
        max_new_tokens=16,
        temperature=0.0,
        top_p=0.95,
        torch_dtype="auto",
        device_map="auto",
        load_in_4bit=False,
        output_path=tmp_path / "predictions.jsonl",
        metrics_output_path=tmp_path / "metrics.json",
        resume=True,
    )

    assert metrics["total"] == 2
    assert metrics["correct"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["average_completion_chars"] > 0


def test_eval_resume_ignores_records_from_other_generation_config(tmp_path: Path):
    output_path = tmp_path / "predictions.jsonl"
    prompt = "What is 2 + 2?"
    output_path.write_text(
        (
            '{"example_index": 0, "prompt_hash": "'
            + baseline.prompt_hash(prompt)
            + '", "model": "dummy", "adapter": null, "split": "test", '
            + '"max_new_tokens": 16, "temperature": 0.0, "top_p": 0.95, '
            + '"use_chat_template": true}\n'
        ),
        encoding="utf-8",
    )

    completed = baseline.read_completed_indices(
        output_path,
        expected_prompt_hashes={0: baseline.prompt_hash(prompt)},
        expected_generation_config={
            "model": "dummy",
            "adapter": None,
            "split": "test",
            "max_new_tokens": 32,
            "temperature": 0.0,
            "top_p": 0.95,
            "use_chat_template": True,
        },
    )

    assert completed == set()
