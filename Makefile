TASKS  ?= crystal_system space_group atom saxs xrd xpdf
TASK   ?= crystal_system
SEED   ?= 0
DEVICE ?= cuda
SOURCE ?= /path/to/CHILI-100K.zip

.PHONY: help setup download test evaluate descriptors couplings internals streams datacard analysis train benchmark verify-benchmark all clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2}'

setup:  ## install dependencies
	pip install -r requirements.txt

download:  ## benchmark data, 21 checkpoints and training records from Hugging Face
	python scripts/download.py --all

test:  ## unit tests
	python -m pytest -q

evaluate:  ## test-set metrics of every checkpoint + results table
	python scripts/evaluate.py --device $(DEVICE)
	python scripts/evaluate.py --tasks space_group --sg-vocab train --device $(DEVICE) --out-dir results/evaluation_sg_trainvocab

descriptors:  ## physical descriptors of the test particles (CPU)
	python experiments/descriptors.py

couplings: descriptors  ## KAN hidden units vs physical descriptors (heatmap)
	python experiments/couplings.py --device $(DEVICE)

internals: descriptors  ## pooling gates and layer probes
	python experiments/internals.py --device $(DEVICE)

streams:  ## stream ablation
	python experiments/stream_ablation.py --device $(DEVICE)

datacard:  ## dataset figures
	python experiments/data_card.py

analysis: couplings internals streams datacard  ## every analysis and figure

train:  ## train one model, e.g. make train TASK=saxs SEED=0
	python scripts/train.py --task $(TASK) --seed $(SEED) --device $(DEVICE)

benchmark:  ## rebuild data/chili100k_benchmark.h5 from the CHILI-100K release
	python scripts/build_benchmark.py --source $(SOURCE)

verify-benchmark:  ## compare data/chili100k_benchmark.h5 with the CHILI-100K release
	python scripts/build_benchmark.py --source $(SOURCE) --verify data/chili100k_benchmark.h5

all: download test evaluate analysis  ## download, test, evaluate and analyse

clean:  ## remove generated results and figures
	rm -rf results/ figures/
