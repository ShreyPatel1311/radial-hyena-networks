"""Hyperparameters for the released checkpoints."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # architecture
    num_layers: int = 2
    node_dim: int = 64
    edge_dim: int = 64
    graph_dim: int = 64
    kan_hidden: int = 64         # 0 -> a single KANLinear with no hidden layer
    kan_grid: int = 5            # spline coefficients per edge = grid + spline_order
    spline_order: int = 3
    dropout: float = 0.1
    graph_in: int = 3            # [log1p(n_atoms), n_species, size_rank]

    # featurisation
    num_rbf_pos: int = 32
    num_rbf_ang: int = 16
    max_k: int = 12              # bonds per atom kept in the edge stream
    rbf_cutoff: float = 6.0      # A
    pos_scale: float = 10.0      # isotropic coordinate scale for the atom task

    # node stream
    inject_pos: bool = True      # additive raw-coordinate embedding

    # optimisation
    epochs: int = 150
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience_lr: int = 8
    patience_es: int = 25

    # data
    subset_per_class: int = 425  # CHILI benchmark protocol


DEFAULT = ModelConfig()
