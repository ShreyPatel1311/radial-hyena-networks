"""Evaluation metrics: weighted / macro F1 and accuracy (classification), MSE (regression)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import CLASSIFICATION


def _forward(model, bt, device, cpu_model=None):
    """Forward pass on `device`; a batch that exhausts GPU memory is run on `cpu_model`."""
    try:
        return model(bt.to(device)).float()
    except torch.cuda.OutOfMemoryError:
        if cpu_model is None:
            raise
        torch.cuda.empty_cache()
        return cpu_model(bt.to("cpu")).float().to(device)


def classification_metrics(y, p):
    """Weighted / macro F1 and accuracy; labels of -1 (outside the model's vocabulary) are
    excluded and counted. Returns (metrics, y_kept, p_kept)."""
    keep = y >= 0
    n_unmap = int((~keep).sum())
    y, p = y[keep], p[keep]
    m = {"weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
         "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
         "acc": float(accuracy_score(y, p)),
         "n_eval": int(len(y)), "n_unmappable": n_unmap}
    return m, y, p


def targets(bt, task):
    return bt.y.view(-1) if (task in CLASSIFICATION and task != "atom") else bt.y


@torch.no_grad()
def evaluate(model, loader, task, device, cpu_model=None, return_raw=False):
    model.eval()
    if cpu_model is not None:
        cpu_model.eval()
    if task in CLASSIFICATION:
        ys, ps = [], []
        for bt in loader:
            y = targets(bt, task)
            out = _forward(model, bt, device, cpu_model)
            ps.append(out.argmax(-1).cpu())
            ys.append(y.cpu())
        m, y, p = classification_metrics(torch.cat(ys).numpy(), torch.cat(ps).numpy())
        return (m, y, p) if return_raw else m
    se, n, errs = 0.0, 0, []
    for bt in loader:
        y = bt.y.to(device)
        pred = _forward(model, bt, device, cpu_model)
        e = (pred - y) ** 2
        errs.append(e.mean(dim=-1).cpu())
        se += e.sum().item()
        n += y.numel()
    m = {"mse": se / n, "n_eval": int(n)}
    return (m, torch.cat(errs).numpy(), None) if return_raw else m


def bootstrap_ci(task, y, p, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap over test graphs (classification: weighted F1 of resampled
    predictions; regression: mean of resampled per-graph squared errors, y = errors)."""
    rng = np.random.default_rng(seed)
    N = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        if task in CLASSIFICATION:
            vals.append(f1_score(y[idx], p[idx], average="weighted", zero_division=0))
        else:
            vals.append(float(y[idx].mean()))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "mean": float(np.mean(vals)),
            "std": float(np.std(vals)), "n_boot": n_boot}
