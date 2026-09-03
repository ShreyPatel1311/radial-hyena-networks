"""Evaluation metrics for the CHILI benchmark.

Classification reports weighted F1, macro F1 and accuracy; regression reports MSE on the
per-sample min-max normalised curves.
"""
from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

CLASSIFICATION_TASKS = ("crystal_system", "space_group", "atom")
REGRESSION_TASKS = ("saxs", "xrd", "xpdf")


def metric_key(task: str) -> str:
    return "weighted_f1" if task in CLASSIFICATION_TASKS else "mse"


def is_better(task: str, a: float, b: float) -> bool:
    return a > b if task in CLASSIFICATION_TASKS else a < b


@torch.no_grad()
def evaluate(model, loader, task, device):
    model.eval()
    if task in CLASSIFICATION_TASKS:
        ys, ps = [], []
        for bt in loader:
            bt = bt.to(device)
            out = model(bt)
            t = bt.y.view(-1) if task != "atom" else bt.y
            ps.append(out.argmax(-1).cpu())
            ys.append(t.cpu())
        y = torch.cat(ys).numpy()
        p = torch.cat(ps).numpy()
        keep = y >= 0                    # -1 marks a label outside the train-only vocabulary
        n_unmap = int((~keep).sum())
        y, p = y[keep], p[keep]
        return {"weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
                "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
                "acc": float(accuracy_score(y, p)),
                "n_eval": int(len(y)), "n_unmappable": n_unmap}
    se = n = 0.0
    for bt in loader:
        bt = bt.to(device)
        e = (model(bt).float() - bt.y) ** 2
        se += e.sum().item()
        n += bt.y.numel()
    return {"mse": se / n, "n_eval": int(n)}


def bootstrap_ci(y, p, task, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval for the headline metric."""
    rng = np.random.default_rng(seed)
    N = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        if task in CLASSIFICATION_TASKS:
            vals.append(f1_score(y[idx], p[idx], average="weighted", zero_division=0))
        else:
            vals.append(float(y[idx].mean()))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "mean": float(np.mean(vals)),
            "std": float(np.std(vals)), "n_boot": n_boot}
