DATA_ZIP ?= data/CHILI-100K.zip
TASKS    ?= crystal_system space_group atom saxs xrd xpdf
CKPT     ?= checkpoints
DEVICE   ?= cuda

.PHONY: help setup data test train evaluate pruning conditioning physics all clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:   ## install dependencies
	pip install -r requirements.txt

data:    ## build the index and cache graph payloads
	python scripts/prepare_data.py --data-zip $(DATA_ZIP)

test:    ## run the test suite
	python -m pytest tests/ -q

train:   ## train every task
	@for t in $(TASKS); do python scripts/train.py --task $$t --seed 0 --device $(DEVICE); done

evaluate:  ## evaluate every checkpoint
	@for t in $(TASKS); do python scripts/evaluate.py --task $$t \
		--checkpoint $(CKPT)/$${t}_seed0.pt --device $(DEVICE); done

pruning:   ## KAN spline pruning curves
	@for t in $(TASKS); do python experiments/kan_pruning.py --task $$t \
		--checkpoint $(CKPT)/$${t}_seed0.pt --device $(DEVICE); done

conditioning:  ## conditioning ablation
	@for t in $(TASKS); do python experiments/conditioning_ablation.py --task $$t \
		--checkpoint $(CKPT)/$${t}_seed0.pt --device $(DEVICE); done

physics:   ## descriptors, operator extraction and the Debye analysis
	python experiments/physical_axes.py --data-zip $(DATA_ZIP)
	python experiments/physics_descriptors.py --split test
	@for t in $(TASKS); do python experiments/spline_operator.py --task $$t \
		--checkpoint $(CKPT)/$${t}_seed0.pt --device $(DEVICE); done
	python experiments/spline_physics.py

all: data train evaluate pruning conditioning physics  ## everything, in order

clean:   ## remove generated results
	rm -rf results/ figures/
