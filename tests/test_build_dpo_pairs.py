import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train" / "build_dpo_pairs.py"
SPEC = importlib.util.spec_from_file_location("build_dpo_pairs_script", SCRIPT_PATH)
assert SPEC is not None
build_dpo_pairs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_dpo_pairs)


def test_dpo_pair_resume_checks_generation_config(tmp_path: Path):
    output_path = tmp_path / "pairs.jsonl"
    prompt = "What is 2 + 2?"
    output_path.write_text(
        (
            '{"example_index": 0, "prompt_hash": "'
            + build_dpo_pairs.prompt_hash(prompt)
            + '", "model": "dummy", "split": "train", "num_completions": 8, '
            + '"max_new_tokens": 384, "temperature": 0.7, "top_p": 0.95, '
            + '"use_chat_template": true, "allow_ties": false, '
            + '"include_gold_chosen": false, "gold_fallback": true}\n'
        ),
        encoding="utf-8",
    )

    completed = build_dpo_pairs.read_completed_indices(
        output_path,
        expected_prompt_hashes={0: build_dpo_pairs.prompt_hash(prompt)},
        expected_generation_config={
            "model": "dummy",
            "split": "train",
            "num_completions": 8,
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "use_chat_template": True,
            "allow_ties": False,
            "include_gold_chosen": False,
            "gold_fallback": True,
        },
    )

    assert completed == set()
