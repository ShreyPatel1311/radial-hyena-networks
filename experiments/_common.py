"""Shared helpers for the analysis scripts: paths, checkpoints, and GPU-with-CPU-fallback."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from radial_hyena.checkpoints import restore, test_loader      # noqa: E402
from radial_hyena.config import TASKS                         # noqa: E402

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DESCRIPTORS = os.path.join("results", "analysis", "descriptors_test.npz")


def checkpoint_path(task, ckpt_dir="checkpoints", seed=0, sg_vocab="train"):
    """Analyses use the seed-0 models; for space group the train-vocabulary head by default."""
    if task == "space_group" and sg_vocab == "train":
        return os.path.join(ckpt_dir, f"space_group_trainvocab_seed{seed}.pt")
    return os.path.join(ckpt_dir, f"{task}_seed{seed}.pt")


def load(task, bench, ckpt_dir="checkpoints", device=DEFAULT_DEVICE, seed=0, sg_vocab="train"):
    """(model, cpu_copy, meta, test loader) for one task."""
    model, meta, cpu = restore(checkpoint_path(task, ckpt_dir, seed, sg_vocab), device, with_cpu_copy=True)
    loader = test_loader(bench, task, meta.get("sg_vocab", "all"))
    return model, cpu, meta, loader


def run(fn, model, cpu_model, bt, device):
    """fn(model, batch) on `device`; a batch that exhausts GPU memory runs on the CPU copy.
    Tensors in the result are returned on the CPU."""
    try:
        out = fn(model, bt.to(device))
    except torch.cuda.OutOfMemoryError:
        if cpu_model is None:
            raise
        torch.cuda.empty_cache()
        out = fn(cpu_model, bt.to("cpu"))
    if isinstance(out, torch.Tensor):
        return out.detach().cpu()
    if isinstance(out, (tuple, list)):
        return type(out)(o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in out)
    if isinstance(out, dict):
        return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
    return out


__all__ = ["ROOT", "TASKS", "DEFAULT_DEVICE", "DESCRIPTORS", "checkpoint_path", "load", "run"]
