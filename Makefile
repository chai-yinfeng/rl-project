.PHONY: install-dev install-quant test inspect-gsm8k baseline-smoke baseline-100 evaluate-baseline grpo-dry-run grpo-smoke pipeline-smoke

PYTHON ?= python
BASELINE_OUTPUT ?= data/processed/gsm8k_baseline_qwen_0_5b_100.jsonl
BASELINE_MODEL ?= Qwen/Qwen2.5-0.5B-Instruct

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-quant:
	$(PYTHON) -m pip install -e ".[dev,quantization]"

test:
	pytest

inspect-gsm8k:
	$(PYTHON) scripts/inspect_gsm8k.py --split test --limit 3

baseline-smoke:
	$(PYTHON) scripts/run_gsm8k_baseline.py \
		--model $(BASELINE_MODEL) \
		--limit 10 \
		--batch-size 1 \
		--max-new-tokens 128 \
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

grpo-dry-run:
	$(PYTHON) scripts/run_grpo_smoke.py --dry-run

grpo-smoke:
	$(PYTHON) scripts/run_grpo_smoke.py --max-train-examples 8 --max-steps 1

pipeline-smoke:
	bash scripts/run_pipeline_smoke.sh
