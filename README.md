# Radial Hyena Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Three-stream cross-gated Hyena network with a Kolmogorov–Arnold (KAN) read-out for
nanoparticle structure–property prediction on **CHILI-100K**, together with the
mechanistic analyses reported in the paper.

Everything runs locally.

## Contents

| path | contents |
|---|---|
| `radial_hyena/` | model, data pipeline, metrics, physics descriptors, operator extraction |
| `scripts/` | data preparation, training, evaluation |
| `experiments/` | KAN pruning, conditioning ablation, spline–physics analyses |
| `tests/` | test suite |

## Installation

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

Python ≥ 3.10, PyTorch ≥ 2.1 with `torch-geometric`. Around 8 GB of VRAM at the default
batch size of 16.

## Data

CHILI-100K is a ~14.5 GB zip archive of HDF5 files. Build the cache once:

```bash
python scripts/prepare_data.py --data-zip /path/to/CHILI-100K.zip
```

```
benchmark subset: 2,975 graphs -> train 2379 / val 298 / test 298
space-group classes (train-only): 151 | test graphs outside vocab: 3
matches the benchmark protocol: True
```

## Training

```bash
python scripts/train.py --task crystal_system --seed 0
```

Tasks: `crystal_system`, `space_group`, `atom`, `saxs`, `xrd`, `xpdf`. Model selection and
the learning-rate schedule use the validation split; each checkpoint stores the weights
and the val/test metrics from the same epoch.

## Evaluation

```bash
python scripts/evaluate.py --task crystal_system --checkpoint checkpoints/crystal_system_seed0.pt
```

Re-measures the test metric and compares it against the value stored in the checkpoint.

## Experiments

```bash
# spline pruning: performance vs fraction of learned splines removed
python experiments/kan_pruning.py --task crystal_system --checkpoint checkpoints/crystal_system_seed0.pt

# conditioning ablation: sequence ordering and filter conditioning, swept separately
python experiments/conditioning_ablation.py --task crystal_system --checkpoint checkpoints/crystal_system_seed0.pt

# spline-physics: read-out Jacobians projected onto the Debye kernel
python experiments/physical_axes.py --data-zip /path/to/CHILI-100K.zip
python experiments/physics_descriptors.py --split test
python experiments/spline_operator.py --task xpdf --checkpoint checkpoints/xpdf_seed0.pt
python experiments/spline_physics.py

# per-unit couplings between hidden units and physical descriptors
python experiments/descriptor_couplings.py
```

`make help` lists shortcuts that loop over all six tasks.
