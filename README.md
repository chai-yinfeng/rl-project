# EECS E6892 RL Project: GRPO for LLM Reasoning

This repository contains the course project for EECS E6892 Reinforcement Learning.

## Goal

Implement and evaluate Group Relative Policy Optimization (GRPO) for small language
model reasoning tasks, with GSM8K as the first benchmark. The project will compare
GRPO against PPO-style RLHF and DPO-style preference optimization where feasible.

## Core Questions

- Does group-relative advantage estimation improve training stability for reasoning?
- How does group size affect variance, sample efficiency, and final accuracy?
- What are the practical tradeoffs between GRPO, PPO, and DPO for small open models?

## Repository Layout

- `configs/`: experiment and model configuration files.
- `data/`: local dataset cache and processed artifacts.
- `docs/`: project planning notes and proposal-derived design decisions.
- `notebooks/`: exploratory analysis and result inspection.
- `reports/`: figures, tables, and final report assets.
- `scripts/`: runnable training, evaluation, and data preparation entrypoints.
- `src/grpo_reasoning/`: reusable project source code.
- `tests/`: focused unit tests for data processing, rewards, and algorithm helpers.

## Initial Milestones

1. Build a supervised/evaluation baseline on GSM8K.
2. Implement reward extraction for math answers.
3. Implement GRPO training with configurable group size.
4. Add PPO/DPO baselines or use well-scoped library baselines.
5. Run controlled experiments and summarize stability, accuracy, and cost.

