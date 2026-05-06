# Experiment Design

## Hardware Assumption

The server target is a single 32GB RTX PRO 4500-class GPU. This is enough for
Qwen2.5-0.5B and Qwen2.5-1.5B LoRA experiments across baseline, DPO, GRPO, and a
compact PPO-style rule-reward trainer. Larger models should use QLoRA first and
short completion lengths.

## Primary Entry Point

Use the Makefile targets for reproducible runs:

```bash
make qwen1_5b-dry-run
make qwen1_5b-test
make qwen1_5b-run
make qwen1_5b-eval
```

Per-method runs follow the same pattern:

```bash
make dpo-1_5b-dry-run
make dpo-1_5b-test
make dpo-1_5b-run
make dpo-1_5b-eval
```

Outputs are grouped under:

```text
experiments/<run_name>/
  config.json
  configs/
    dpo_train.json
    grpo_train.json
    ppo_train.json
  data/
    baseline_predictions.jsonl
    dpo_pairs.jsonl
    dpo_predictions.jsonl
    grpo_predictions.jsonl
    ppo_predictions.jsonl
  logs/
  metrics/
    baseline.json
    dpo.json
    grpo.json
    ppo.json
  models/
    dpo/
    grpo/
    ppo/
```

## Recommended Run Matrix

Start with the 0.5B config to verify the complete pipeline:

| Run | Model | Eval | DPO pairs | GRPO | PPO-style | Purpose |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0.5B test | Qwen2.5-0.5B-Instruct | 20 | 20 x 2 | 8 examples / 1 step | 8 examples / 1 step | Validate complete framework |
| 0.5B full | Qwen2.5-0.5B-Instruct | 100 | 200 x 4 | 128 examples / 50 steps | 128 examples / 50 steps | Small comparison |

Then run the 1.5B config as the main course-project experiment:

| Run | Model | Eval | DPO pairs | GRPO | PPO-style | Purpose |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1.5B test | Qwen2.5-1.5B-Instruct | 20 | 20 x 2 | 8 examples / 1 step | 8 examples / 1 step | Server validation |
| 1.5B full | Qwen2.5-1.5B-Instruct | 200 | 500 x 4 | 256 examples / 100 steps | 256 examples / 100 steps | Main comparison |

Optional extensions if time remains:

| Variant | Change | Reason |
| --- | --- | --- |
| GRPO-KL | `beta=0.001` or `0.01` | Measure memory/time cost of KL reference control |
| GRPO-G4 | `num_generations=4` | Test stronger group-relative signal |
| DPO-1k | 1000 train prompts x 4 completions | Test pair count scaling |
| Eval-500 | eval limit 500 | More stable final accuracy estimate |

## Metrics to Record

Task metrics:

- exact-match accuracy
- invalid answer rate
- average completion length
- format-following rate if added later

Training efficiency:

- wall-clock elapsed seconds
- step time
- peak CUDA allocated memory
- peak CUDA reserved memory
- train examples and generation budget

DPO-specific:

- chosen/rejected pair count
- kept-pair rate from sampled prompts
- `rewards/chosen`, `rewards/rejected`, margin, and accuracy from trainer logs

GRPO-specific:

- reward mean and reward std
- completion length
- KL when `beta > 0`
- effect of `num_generations`

PPO-style-specific:

- reward mean and reward std
- invalid rate
- approximate KL from old policy
- reference KL when `kl_beta > 0`
- policy loss

## Downloading Results

After runs complete, package the results directory:

```bash
make package-results
```

This creates a timestamped `results_*.tar.gz` archive that can be downloaded from
VS Code Remote SSH. The packaging command excludes intermediate checkpoint folders
by default but keeps final outputs, metrics, configs, prediction files, and logs.

If you want every checkpoint too, download the raw `experiments/` directory instead.

## Interpretation Plan

The report should not claim SOTA. The defensible claim is:

```text
Under a fixed verifiable GSM8K reward and small-model budget, DPO has the simplest
offline preference pipeline, GRPO uses online grouped samples to create relative
advantages, and PPO-style training has the heaviest rollout/KL bookkeeping.
```

The core comparison table should include:

| Method | Accuracy | Invalid rate | Avg length | Train time | Peak memory | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | | | | | | no training |
| DPO | | | | | | offline pairs |
| GRPO | | | | | | online group reward |
| PPO-style | | | | | | rollout + clipped objective |
