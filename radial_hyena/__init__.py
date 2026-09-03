"""Radial Hyena Networks: three-stream cross-gated Hyena with a KAN read-out."""
__version__ = "1.0.0"

from .config import ModelConfig, DEFAULT
from .kan import KAN, KANLinear
from .model import RadialHyenaNet, RadialNodeStream, EdgeHyenaStream, GraphStream
from .metrics import evaluate, CLASSIFICATION_TASKS, metric_key
from .runner import load_task, build_model, make_loader, reproduction_gate

__all__ = ["ModelConfig", "DEFAULT", "KAN", "KANLinear", "RadialHyenaNet",
           "RadialNodeStream", "EdgeHyenaStream", "GraphStream",
           "evaluate", "CLASSIFICATION_TASKS", "metric_key",
           "load_task", "build_model", "make_loader", "reproduction_gate"]
