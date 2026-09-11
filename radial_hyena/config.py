"""Hyper-parameters of the published Radial Hyena runs and the CHILI-100K task table."""
from __future__ import annotations

from dataclasses import asdict, dataclass

TASKS = ("crystal_system", "space_group", "atom", "saxs", "xrd", "xpdf")
CLASSIFICATION = ("crystal_system", "space_group", "atom")
SCATTERING = {"saxs": "SAXS", "xrd": "XRD", "xpdf": "xPDF"}
CRYSTAL_SYSTEMS = ("Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal",
                   "Trigonal", "Hexagonal", "Cubic")
NICE = {"crystal_system": "Crystal system", "space_group": "Space group",
        "atom": "Atom type", "saxs": "SAXS", "xrd": "XRD", "xpdf": "xPDF"}

# fixed output widths; space group is 230 (all crystallographic groups) unless the
# train-split vocabulary is requested
OUT_DIM = {"crystal_system": 7, "space_group": 230, "atom": 118,
           "saxs": 300, "xrd": 580, "xpdf": 6000}

SUBSET_PER_CLASS = 425      # benchmark subset: graphs per crystal system
SPLIT_SEED = 42             # subset selection and the 80/10/10 split
TRAIN_EVAL_SEED = 12345     # fixed train slice evaluated each epoch for monitoring


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int = 2
    node_dim: int = 64
    edge_dim: int = 64
    graph_dim: int = 64
    graph_in: int = 3               # [log1p(n_atoms), n_species, size_rank]
    kan_hidden: int = 64
    kan_grid: int = 5
    spline_order: int = 3
    dropout: float = 0.1            # KAN read-out only
    inject_pos: bool = True         # additive embedding of centred coordinates (node stream)
    num_rbf_pos: int = 32
    num_rbf_ang: int = 16
    max_k: int = 12                 # bonds per atom in the edge-stream sequence
    filter_hidden: int = 64
    rbf_cutoff: float = 6.0         # Å
    pos_scale: float = 10.0         # atom task: centred coordinates / pos_scale

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 150
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lr_factor: float = 0.5          # ReduceLROnPlateau on the validation metric
    patience_lr: int = 8
    min_lr: float = 1e-6
    patience_es: int = 25           # early stopping on the validation metric
    grad_clip: float = 1.0
    node_drop: float = 0.2          # training-time atom dropout (train loader only)
    num_workers: int = 2

    def to_dict(self):
        return asdict(self)


MODEL = ModelConfig()
TRAIN = TrainConfig()


def level_of(task: str) -> str:
    return "node" if task == "atom" else "graph"


def metric_key(task: str) -> str:
    return "weighted_f1" if task in CLASSIFICATION else "mse"
