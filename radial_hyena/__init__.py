"""Radial Hyena networks for property prediction of nanomaterials (CHILI-100K)."""
from .config import MODEL, TRAIN, TASKS, ModelConfig, TrainConfig
from .model import RadialHyena, count_parameters

__all__ = ["RadialHyena", "count_parameters", "ModelConfig", "TrainConfig",
           "MODEL", "TRAIN", "TASKS"]
__version__ = "1.0.0"
