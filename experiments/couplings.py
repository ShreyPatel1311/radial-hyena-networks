#!/usr/bin/env python3
"""Partial Spearman coupling between the KAN's 64 hidden units and physical descriptors.

For every task (seed-0 model, test split) the 64-d input of the KAN's second layer is
captured per test graph (atom task: per-atom rows, averaged within each graph). Each unit
is correlated with each of 14 physical descriptors by partial Spearman correlation:
all variables are rank-transformed and the three graph-level inputs of the model
(log1p n_atoms, n_species, size_rank) are regressed out of both sides.

Outputs: results/analysis/couplings.json and figures/descriptor_couplings.png
Requires results/analysis/descriptors_test.npz (experiments/descriptors.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DEFAULT_DEVICE, DESCRIPTORS, TASKS, checkpoint_path, load, run   # noqa: E402
from radial_hyena.config import NICE                                        # noqa: E402
from radial_hyena.data import load_benchmark                                # noqa: E402

CONFOUND = ["log_n_atoms", "n_species", "size_rank"]
SKIP = {"crystal_system", "space_group", "size_rank", "n_atoms", "log_n_atoms", "n_species"}
MAX_ROWS = 20000       # node-level task: rows kept (evenly spaced) before per-graph averaging


@torch.no_grad()
def hidden_units(model, cpu_model, loader, device, max_rows=MAX_ROWS):
    """Input of KAN block 1 (the 64 hidden units) for every row, with each row's graph id."""
    rows, gids, g0 = [], [], 0

    def fn(m, bt):
        cache = {}
        h = m.kan.blocks[1].register_forward_hook(
            lambda mod, inp, out: cache.__setitem__("x", inp[0].reshape(-1, mod.in_features)))
        try:
            m(bt)
        finally:
            h.remove()
        return cache["x"].float()

    model.eval()
    for bt in loader:
        x = run(fn, model, cpu_model, bt, device)
        batch = bt.batch.cpu().numpy()
        nb = int(batch.max()) + 1
        gids.append(np.arange(g0, g0 + nb) if x.shape[0] == nb else g0 + batch)
        rows.append(x)
        g0 += nb
    X, g = torch.cat(rows).numpy(), np.concatenate(gids)
    if X.shape[0] > max_rows:
        sel = np.linspace(0, X.shape[0] - 1, max_rows).astype(int)
        X, g = X[sel], g[sel]
    return X, g, g0


def per_graph(x, gids, n_graphs):
    if x.shape[0] == n_graphs:
        return x
    agg, cnt = np.zeros((n_graphs, x.shape[1])), np.zeros(n_graphs)
    np.add.at(agg, gids, x)
    np.add.at(cnt, gids, 1)
    return agg / np.maximum(cnt, 1)[:, None]


def _zc(a):
    a = a - a.mean(0, keepdims=True)
    s = a.std(0, keepdims=True)
    return a / np.where(s > 1e-12, s, 1.0)


def _rank(a):
    r = rankdata(a).astype(float)
    return (r - r.mean()) / max(r.std(), 1e-12)


def _resid(y, Z):
    A = np.column_stack([np.ones(len(y)), _zc(Z)])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ b


def partial_spearman(x, y, Z):
    """Rank-transform x, y and the confounds, then correlate the residuals."""
    xr, yr = _rank(x), _rank(y)
    Zr = np.column_stack([_rank(Z[:, k]) for k in range(Z.shape[1])])
    rx, ry = _resid(xr, Zr), _resid(yr, Zr)
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.mean((rx / sx) * (ry / sy)))


def coupling_matrix(X, desc):
    """|partial Spearman| between every unit (rows of X = test graphs) and every descriptor."""
    allnames = [str(n) for n in desc["names"]]
    names = [d for d in allnames if d not in SKIP]
    V = desc["values"][:, [allnames.index(d) for d in names]]
    Z = desc["values"][:, [allnames.index(c) for c in CONFOUND]]
    C = np.zeros((X.shape[1], len(names)))
    for k in range(len(names)):
        m = np.isfinite(V[:, k])
        for j in range(X.shape[1]):
            xv = X[m, j]
            if xv.std() < 1e-12:
                continue
            C[j, k] = np.nan_to_num(partial_spearman(xv, V[m, k], Z[m]))
    return np.abs(C), names


def heatmap(mats, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = [t for t in TASKS if t in mats]
    fig, axs = plt.subplots(3, 2, figsize=(12.5, 13.5))
    im = None
    for ax, t in zip(axs.ravel(), tasks):
        C = mats[t]
        order = np.argsort(-C.max(1))
        im = ax.imshow(C[order].T, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlabel("hidden unit (sorted by peak coupling)", fontsize=12)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_title(NICE[t], fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.90, 1])
    cax = fig.add_axes([0.925, 0.18, 0.018, 0.64])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("partial |Spearman|", fontsize=12)
    cb.ax.tick_params(labelsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--data", default="data/chili100k_benchmark.h5")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--sg-vocab", choices=["all", "train"], default="train")
    ap.add_argument("--descriptors", default=DESCRIPTORS)
    ap.add_argument("--out-dir", default=os.path.join("results", "analysis"))
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from couplings.json")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.figdir, exist_ok=True)
    path = os.path.join(a.out_dir, "couplings.json")
    if a.plot_only:
        doc = json.load(open(path))
        mats = {t: np.array(v["matrix_units_x_descriptors"]) for t, v in doc["tasks"].items()}
        heatmap(mats, doc["descriptors"], os.path.join(a.figdir, "descriptor_couplings.png"))
        print(f"wrote {a.figdir}/descriptor_couplings.png ({', '.join(t for t in TASKS if t in mats)})")
        return
    bench = load_benchmark(a.data)
    desc = np.load(a.descriptors, allow_pickle=True)
    assert list(desc["test_index"]) == bench.split["test"], "descriptors are not for this test split"

    # tasks from an earlier run are kept, so the tasks can be computed in separate runs
    prev = json.load(open(path))["tasks"] if os.path.exists(path) else {}
    mats = {t: np.array(v["matrix_units_x_descriptors"]) for t, v in prev.items() if t not in a.tasks}
    out = {t: v for t, v in prev.items() if t not in a.tasks}
    names = None
    for task in a.tasks:
        model, cpu, meta, loader = load(task, bench, a.ckpt_dir, a.device, sg_vocab=a.sg_vocab)
        X, g, n = hidden_units(model, cpu, loader, a.device)
        C, names = coupling_matrix(per_graph(X, g, n), desc)
        mats[task] = C
        peak = C.max(0)
        out[task] = {"checkpoint": os.path.basename(checkpoint_path(task, a.ckpt_dir, 0, a.sg_vocab)),
                     "peak_abs_partial_spearman": dict(zip(names, map(float, peak))),
                     "units_above_0.4": int((C.max(1) > 0.4).sum()),
                     "matrix_units_x_descriptors": C.tolist()}
        top = sorted(zip(names, peak), key=lambda kv: -kv[1])[:3]
        print(f"{task:15s} strongest: " + ", ".join(f"{k} {v:.3f}" for k, v in top), flush=True)
        del model, cpu
        torch.cuda.empty_cache() if a.device.startswith("cuda") else None
    out = {t: out[t] for t in TASKS if t in out}
    with open(path, "w") as f:
        json.dump({"descriptors": names, "confounds": CONFOUND, "tasks": out}, f, indent=2)
    heatmap(mats, names, os.path.join(a.figdir, "descriptor_couplings.png"))
    print(f"wrote {a.figdir}/descriptor_couplings.png")


if __name__ == "__main__":
    main()
