.PHONY: lock sync sync-quant pip-install-dev pip-install-quant pip-test test inspect-gsm8k baseline-smoke baseline-100 evaluate-baseline dpo-pairs-dry-run dpo-pairs-small dpo-train-dry-run dpo-train-small dpo-smoke grpo-dry-run grpo-lite-dry-run grpo-smoke grpo-lite pipeline-smoke

UV ?= uv
RUN ?= $(UV) run
PYTHON ?= $(RUN) python

BASELINE_OUTPUT ?= data/processed/gsm8k_baseline_qwen_0_5b_100.jsonl
BASELINE_MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
DPO_PAIRS_OUTPUT ?= data/processed/gsm8k_dpo_pairs_qwen_0_5b_train200_k4.jsonl
DPO_MODEL ?= $(BASELINE_MODEL)

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

baseline-smoke:
	$(PYTHON) scripts/run_gsm8k_baseline.py \
		--model $(BASELINE_MODEL) \
		--limit 10 \
		--batch-size 1 \
		--max-new-tokens 512 \
		--torch-dtype float16 \
		--output data/processed/gsm8k_baseline_smoke.jsonl \
		--resume

baseline-100:
	$(PYTHON) scripts/run_gsm8k_baseline.py \
		--model $(BASELINE_MODEL) \
		--limit 100 \
		--batch-size 1 \
		--max-new-tokens 256 \
		--torch-dtype float16 \
		--output $(BASELINE_OUTPUT) \
		--resume

evaluate-baseline:
	$(PYTHON) scripts/evaluate_predictions.py $(BASELINE_OUTPUT)

dpo-pairs-dry-run:
	$(PYTHON) scripts/build_dpo_pairs.py \
		--model $(DPO_MODEL) \
		--limit 200 \
		--num-completions 4 \
		--output $(DPO_PAIRS_OUTPUT) \
		--dry-run

dpo-pairs-small:
	$(PYTHON) scripts/build_dpo_pairs.py \
		--model $(DPO_MODEL) \
		--limit 200 \
		--num-completions 4 \
		--max-new-tokens 128 \
		--temperature 0.7 \
		--top-p 0.95 \
		--torch-dtype float16 \
		--output $(DPO_PAIRS_OUTPUT) \
		--resume

dpo-train-dry-run:
	$(PYTHON) scripts/run_dpo_train.py --dry-run

dpo-train-small:
	$(PYTHON) scripts/run_dpo_train.py

dpo-smoke:
	bash scripts/run_dpo_smoke.sh

grpo-dry-run:
	$(PYTHON) scripts/run_grpo_smoke.py --dry-run

grpo-lite-dry-run:
	$(PYTHON) scripts/run_grpo_smoke.py --config configs/train/grpo_lite_8gb.json --dry-run

grpo-smoke:
	$(PYTHON) scripts/run_grpo_smoke.py --max-train-examples 8 --max-steps 1

grpo-lite:
	$(PYTHON) scripts/run_grpo_smoke.py --config configs/train/grpo_lite_8gb.json

pipeline-smoke:
	bash scripts/run_pipeline_smoke.sh
