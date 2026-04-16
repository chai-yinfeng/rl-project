# Project Plan

## Proposal Summary

The proposal studies reinforcement learning for LLM reasoning through Group Relative
Policy Optimization (GRPO). The main benchmark is GSM8K, and the intended comparison
set is PPO-based RLHF and DPO. The expected output is a working implementation,
empirical comparisons, and analysis of relative reward optimization.

## Recommended Scope

Use one small instruction-capable model as the primary policy model. Start with GSM8K
answer accuracy and answer-format reward extraction before adding more benchmarks.
Keep PPO and DPO comparisons practical: if full PPO/DPO training is too expensive,
use a smaller controlled baseline or cite the limitation clearly.

## Directory Plan

```text
configs/
  model/
  train/
  eval/
data/
  raw/
  processed/
docs/
notebooks/
reports/
scripts/
src/grpo_reasoning/
  algorithms/
  data/
  evaluation/
  models/
  rewards/
  training/
  utils/
tests/
```

## Work Packages

1. Environment and reproducibility
   - Pin Python dependencies.
   - Add a smoke-test command for imports and dataset loading.
   - Decide device target: local CPU/MPS, local CUDA, or Colab/HPC GPU.

2. Dataset and evaluation
   - Load GSM8K through `datasets`.
   - Normalize prompts and final answers.
   - Implement exact-match answer extraction.
   - Track accuracy, invalid answer rate, response length, and inference cost.

3. Baseline model
   - Pick a small model that can run within the available compute budget.
   - Run zero-shot and few-shot chain-of-thought baselines.
   - Save predictions for later comparison.

4. Reward design
   - Implement correctness reward from final numeric answer.
   - Add optional format reward for reasoning/final-answer structure.
   - Log reward distribution before training.

5. GRPO implementation
   - Generate a group of completions per prompt.
   - Compute group-relative normalized advantages.
   - Apply clipped policy-gradient loss with KL control against a reference model.
   - Make group size configurable.

6. Baselines
   - DPO: use synthetic or generated preference pairs if no human preference data exists.
   - PPO: use TRL PPO where feasible, or document compute limits and compare against
     supervised/evaluation baselines.

7. Experiments
   - Sweep group size, learning rate, and KL coefficient.
   - Compare training stability through reward variance, loss curves, and KL.
   - Report final GSM8K accuracy and sample efficiency.

8. Final report
   - Include method description, experiment setup, ablations, and failure cases.
   - Discuss where GRPO helps and where it is limited by compute or reward quality.

## Proposal Improvements

- Specify the exact model family and size target.
- Clarify compute assumptions and fallback plans if PPO is too expensive.
- Define metrics beyond accuracy: KL, reward variance, invalid answer rate, and cost.
- Clarify whether DPO data will come from generated pairs, public preference data, or
  a simplified baseline.
- Add a risk section covering reward hacking, unstable generation, and hardware limits.

