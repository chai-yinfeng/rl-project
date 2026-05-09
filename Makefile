.PHONY: lock sync sync-quant pip-install-dev pip-install-quant pip-test test inspect-gsm8k package-results \
	qwen0_5b-dry-run qwen0_5b-test qwen0_5b-run qwen0_5b-eval qwen0_5b-all \
	qwen1_5b-dry-run qwen1_5b-test qwen1_5b-run qwen1_5b-eval qwen1_5b-all \
	math-1_5b-dry-run math-1_5b-test math-1_5b-run math-1_5b-eval \
	grpo-gs2-1_5b-dry-run grpo-gs2-1_5b-run grpo-gs2-1_5b-eval grpo-gs2-1_5b-all \
	grpo-gs4-1_5b-dry-run grpo-gs4-1_5b-run grpo-gs4-1_5b-eval grpo-gs4-1_5b-all \
	grpo-gs8-1_5b-dry-run grpo-gs8-1_5b-run grpo-gs8-1_5b-eval grpo-gs8-1_5b-all \
	baseline-0_5b-dry-run baseline-0_5b-test baseline-0_5b-run baseline-0_5b-eval \
	dpo-0_5b-dry-run dpo-0_5b-test dpo-0_5b-run dpo-0_5b-eval \
	grpo-0_5b-dry-run grpo-0_5b-test grpo-0_5b-run grpo-0_5b-eval \
	ppo-0_5b-dry-run ppo-0_5b-test ppo-0_5b-run ppo-0_5b-eval \
	baseline-1_5b-dry-run baseline-1_5b-test baseline-1_5b-run baseline-1_5b-eval \
	dpo-1_5b-dry-run dpo-1_5b-test dpo-1_5b-run dpo-1_5b-eval \
	grpo-1_5b-dry-run grpo-1_5b-test grpo-1_5b-run grpo-1_5b-eval \
	ppo-1_5b-dry-run ppo-1_5b-test ppo-1_5b-run ppo-1_5b-eval

UV ?= uv
RUN ?= $(UV) run
PYTHON ?= $(RUN) python

QWEN0_TEST_CONFIG := configs/experiments/gsm8k_qwen0_5b_test.json
QWEN0_FULL_CONFIG := configs/experiments/gsm8k_qwen0_5b_full.json
QWEN1_TEST_CONFIG := configs/experiments/gsm8k_qwen1_5b_test.json
QWEN1_FULL_CONFIG := configs/experiments/gsm8k_qwen1_5b_full.json
QWEN1_GRPO_GS2_CONFIG := configs/experiments/gsm8k_qwen1_5b_grpo_gs2.json
QWEN1_GRPO_GS4_CONFIG := configs/experiments/gsm8k_qwen1_5b_grpo_gs4.json
QWEN1_GRPO_GS8_CONFIG := configs/experiments/gsm8k_qwen1_5b_grpo_gs8.json
RESULTS_DIR ?= experiments

lock:
	$(UV) lock

sync:
	$(UV) sync --extra dev

sync-quant:
	$(UV) sync --extra dev --extra quantization

pip-install-dev:
	python -m pip install -e ".[dev]"

pip-install-quant:
	python -m pip install -e ".[dev,quantization]"

pip-test:
	python -m pytest

test:
	$(PYTHON) -m pytest

inspect-gsm8k:
	$(PYTHON) scripts/inspect_gsm8k.py --split test --limit 3

package-results:
	bash scripts/package_results.sh $(RESULTS_DIR)

qwen0_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage all --dry-run

qwen0_5b-test:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_TEST_CONFIG) --stage all

qwen0_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage run

qwen0_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage eval

qwen0_5b-all:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage all

qwen1_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage all --dry-run

qwen1_5b-test:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_TEST_CONFIG) --stage all

qwen1_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage run

qwen1_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage eval

qwen1_5b-all:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage all

math-1_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage math --dry-run

math-1_5b-test:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_TEST_CONFIG) --stage math

math-1_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage math-eval

math-1_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage math

grpo-gs2-1_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_GRPO_GS2_CONFIG) --stage grpo --dry-run

grpo-gs2-1_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_GRPO_GS2_CONFIG) --stage grpo-run

grpo-gs2-1_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_GRPO_GS2_CONFIG) --stage grpo-eval

grpo-gs2-1_5b-all:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_GRPO_GS2_CONFIG) --stage grpo

grpo-gs4-1_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_GRPO_GS4_CONFIG) --stage grpo --dry-run

grpo-gs4-1_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_GRPO_GS4_CONFIG) --stage grpo-run

grpo-gs4-1_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_GRPO_GS4_CONFIG) --stage grpo-eval

grpo-gs4-1_5b-all:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_GRPO_GS4_CONFIG) --stage grpo

grpo-gs8-1_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_GRPO_GS8_CONFIG) --stage grpo --dry-run

grpo-gs8-1_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_GRPO_GS8_CONFIG) --stage grpo-run

grpo-gs8-1_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_GRPO_GS8_CONFIG) --stage grpo-eval

grpo-gs8-1_5b-all:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_GRPO_GS8_CONFIG) --stage grpo

baseline-0_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage baseline --dry-run

baseline-0_5b-test:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_TEST_CONFIG) --stage baseline

baseline-0_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage baseline

baseline-0_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage baseline

dpo-0_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage dpo --dry-run

dpo-0_5b-test:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_TEST_CONFIG) --stage dpo

dpo-0_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage dpo-run

dpo-0_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage dpo-eval

grpo-0_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage grpo --dry-run

grpo-0_5b-test:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_TEST_CONFIG) --stage grpo

grpo-0_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage grpo-run

grpo-0_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage grpo-eval

ppo-0_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage ppo --dry-run

ppo-0_5b-test:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_TEST_CONFIG) --stage ppo

ppo-0_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage ppo-run

ppo-0_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN0_FULL_CONFIG) --stage ppo-eval

baseline-1_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage baseline --dry-run

baseline-1_5b-test:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_TEST_CONFIG) --stage baseline

baseline-1_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage baseline

baseline-1_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage baseline

dpo-1_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage dpo --dry-run

dpo-1_5b-test:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_TEST_CONFIG) --stage dpo

dpo-1_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage dpo-run

dpo-1_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage dpo-eval

grpo-1_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage grpo --dry-run

grpo-1_5b-test:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_TEST_CONFIG) --stage grpo

grpo-1_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage grpo-run

grpo-1_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage grpo-eval

ppo-1_5b-dry-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage ppo --dry-run

ppo-1_5b-test:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_TEST_CONFIG) --stage ppo

ppo-1_5b-run:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage ppo-run

ppo-1_5b-eval:
	$(PYTHON) scripts/run_experiment.py --config $(QWEN1_FULL_CONFIG) --stage ppo-eval
