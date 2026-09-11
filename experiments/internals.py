#!/usr/bin/env python3
"""Pooling gates and layer probes of the trained models (seed 0, test split).

A. Pooling gates   The node-pooling softmax weight of every atom, scaled by the particle's
                   atom count (1.0 = uniform attention), binned into core / mid / surface
                   thirds of the normalised radius, and its Spearman correlation with the
                   atom's normalised radius, coordination number and atomic number.
                   Graph-level tasks only (the atom task has no pooling gate).
C. Layer probes    Cross-validated ridge-regression R^2 for each physical descriptor from the
                   graph-mean node representation after the embedding and after each Hyena
                   layer, and from the pooled node representation.

Outputs: results/analysis/internals_<task>.json, figures/pooling_gates.png,
         figures/layer_probes.png. Requires results/analysis/descriptors_test.npz.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DEFAULT_DEVICE, DESCRIPTORS, TASKS, load, run          # noqa: E402
from radial_hyena.config import NICE                                       # noqa: E402
from radial_hyena.data import load_benchmark                               # noqa: E402
from radial_hyena.model import scatter_mean, scatter_softmax, scatter_sum  # noqa: E402

SKIP = {"crystal_system", "space_group", "size_rank", "n_atoms", "log_n_atoms", "n_species"}
DEPTHS = ["embed", "layer1", "layer2", "pooled"]
DEPTH_LABEL = ["embedding", "Hyena layer 1", "Hyena layer 2", "pooled"]


# --------------------------------------------------------------------- one pass over the test set
def _internals(m, bt):
    """Particle-level features at every depth, plus per-atom gate rows
    [gate weight x N, r/r_max, coordination, Z, graph]."""
    h_v, _, _, batch, B, layers = m.trunk(bt, keep_layers=True)
    out = {"embed": scatter_mean(layers[0], batch, B)}
    for i, h in enumerate(layers[1:]):
        out[f"layer{i + 1}"] = scatter_mean(h, batch, B)
    nv = scatter_softmax(m.node_gate(h_v), batch, B)
    out["pooled"] = scatter_sum(h_v * nv, batch, B)
    cen = scatter_mean(bt.pos, batch, B)
    r = (bt.pos - cen[batch]).norm(dim=-1)
    rmax = torch.zeros(B, device=r.device).scatter_reduce(0, batch, r, reduce="amax",
                                                          include_self=False).clamp(min=1e-6)
    rn = (r / rmax[batch]).clamp(0, 1)
    N = h_v.shape[0]
    src = bt.edge_index[0][bt.edge_index[0] != bt.edge_index[1]]
    deg = torch.bincount(src, minlength=N).float()
    cnt = torch.bincount(batch, minlength=B).float()
    out["gates"] = torch.stack([nv.squeeze(-1) * cnt[batch], rn, deg, bt.z.float(), batch.float()], -1)
    return out


@torch.no_grad()
def collect(model, cpu_model, loader, device, max_gate_atoms=64 * 4000):
    """Depth features for every test particle; gate rows for the first batches until more
    than `max_gate_atoms` atoms are collected."""
    model.eval(); cpu_model.eval()
    feats, gates, n_gate = {}, [], 0
    for bt in loader:
        o = run(_internals, model, cpu_model, bt, device)
        if n_gate <= max_gate_atoms:
            gates.append(o["gates"])
            n_gate += o["gates"].shape[0]
        for k in DEPTHS:
            feats.setdefault(k, []).append(o[k])
    return {k: torch.cat(v).numpy() for k, v in feats.items()}, torch.cat(gates).numpy()


# --------------------------------------------------------------------- A. pooling gates
def pooling_gates(M):
    out = {"n_atoms": int(M.shape[0])}
    for j, nm in [(1, "radius_norm"), (2, "coordination"), (3, "atomic_number")]:
        m = np.isfinite(M[:, 0]) & np.isfinite(M[:, j])
        out[f"gate_vs_{nm}"] = float(spearmanr(M[m, 0], M[m, j]).statistic)
    w = M[:, 0]
    out["gate_weight_mean_x_natoms"] = float(w.mean())
    for lo, hi, nm in [(0.0, 0.33, "core"), (0.33, 0.66, "mid"), (0.66, 1.01, "surface")]:
        sel = (M[:, 1] >= lo) & (M[:, 1] < hi)
        out[f"mean_gate_{nm}"] = float(w[sel].mean()) if sel.any() else None
    return out


# --------------------------------------------------------------------- C. layer probes
def layer_probes(F, desc):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score
    names = [str(x) for x in desc["names"]]
    V = desc["values"]
    out = {}
    for depth, X in F.items():
        n = min(X.shape[0], V.shape[0])
        Xs = (X[:n] - X[:n].mean(0)) / (X[:n].std(0) + 1e-9)
        row = {}
        for j, nm in enumerate(names):
            if nm in SKIP:
                continue
            y = V[:n, j]
            m = np.isfinite(y)
            if m.sum() < 50:
                continue
            ys = (y[m] - y[m].mean()) / (y[m].std() + 1e-9)
            r2 = cross_val_score(RidgeCV(alphas=np.logspace(-2, 4, 13)), Xs[m], ys,
                                 cv=5, scoring="r2").mean()
            row[nm] = float(r2)
        out[depth] = row
    return out


# --------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _sublabel(ax, i, x=-0.12, y=1.02):
    ax.text(x, y, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")


def fig_gates(res, path):
    plt = _plt()
    col = dict(zip(TASKS, plt.cm.tab10.colors))
    tasks = [t for t in TASKS if t in res and isinstance(res[t].get("pooling_gates"), dict)]
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 8.4))
    w = 0.8 / len(tasks)
    for i, t in enumerate(tasks):
        g = res[t]["pooling_gates"]
        v = [g.get(f"mean_gate_{z}") or np.nan for z in ("core", "mid", "surface")]
        axs[0].bar(np.arange(3) + i * w - 0.4 + w / 2, v, w, label=NICE[t], color=col[t])
    axs[0].axhline(1.0, color="k", lw=.8, ls=":")
    axs[0].set_xticks(range(3)); axs[0].set_xticklabels(["core", "mid", "surface"])
    axs[0].set_ylabel("mean gate weight $\\times$ N atoms")
    axs[0].legend(fontsize=7); axs[0].grid(alpha=.25, axis="y")
    keys = ["gate_vs_radius_norm", "gate_vs_coordination", "gate_vs_atomic_number"]
    for i, t in enumerate(tasks):
        g = res[t]["pooling_gates"]
        axs[1].bar(np.arange(3) + i * w - 0.4 + w / 2, [g.get(k, np.nan) for k in keys], w, color=col[t])
    axs[1].axhline(0, color="k", lw=.8)
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["radius", "coordination", "atomic number"])
    axs[1].set_ylabel("Spearman(gate weight, quantity)")
    axs[1].grid(alpha=.25, axis="y")
    for i, ax in enumerate(axs):
        _sublabel(ax, i)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig_probes(res, path):
    plt = _plt()
    tasks = [t for t in TASKS if t in res and isinstance(res[t].get("layer_probes"), dict)]
    nrow = (len(tasks) + 1) // 2
    fig, axs = plt.subplots(nrow, 2, figsize=(9.0, 3.5 * nrow), squeeze=False)
    for ax, t in zip(axs.ravel(), tasks):
        P = res[t]["layer_probes"]
        for d in sorted(P.get("pooled", {}), key=lambda d: -P["pooled"].get(d, 0))[:8]:
            ax.plot(range(len(DEPTHS)), [P.get(dep, {}).get(d, np.nan) for dep in DEPTHS],
                    "-o", ms=4, label=d)
        ax.set_xticks(range(len(DEPTHS)))
        ax.set_xticklabels(DEPTH_LABEL, rotation=20, ha="right", fontsize=7.5)
        ax.set_ylabel("probe $R^2$"); ax.set_ylim(-0.05, 1.0)
        ax.set_title(NICE[t], fontsize=10); ax.grid(alpha=.25)
        ax.legend(fontsize=6, ncol=2)
    for ax in axs.ravel()[len(tasks):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


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
    ap.add_argument("--plot-only", action="store_true", help="redraw figures from saved JSON")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.figdir, exist_ok=True)

    if not a.plot_only:
        bench = load_benchmark(a.data)
        desc = np.load(a.descriptors, allow_pickle=True)
        assert list(desc["test_index"]) == bench.split["test"], "descriptors are not for this test split"
        for task in a.tasks:
            model, cpu, meta, loader = load(task, bench, a.ckpt_dir, a.device, sg_vocab=a.sg_vocab)
            F, M = collect(model, cpu, loader, a.device)
            res = {"task": task, "recorded_test": meta.get("test"), "out_dim": meta["out_dim"]}
            if task != "atom":
                res["pooling_gates"] = pooling_gates(M)
            res["layer_probes"] = layer_probes(F, desc)
            with open(os.path.join(a.out_dir, f"internals_{task}.json"), "w") as f:
                json.dump(res, f, indent=2)
            g = res.get("pooling_gates")
            print(f"{task:15s} " + (f"gates core/mid/surface {g['mean_gate_core']:.2f}/{g['mean_gate_mid']:.2f}/"
                                    f"{g['mean_gate_surface']:.2f}  " if g else "")
                  + "probe R2 (pooled, best): "
                  + ", ".join(f"{k} {v:.2f}" for k, v in sorted(res["layer_probes"]["pooled"].items(),
                                                                key=lambda kv: -kv[1])[:2]), flush=True)
            del model, cpu
            torch.cuda.empty_cache() if a.device.startswith("cuda") else None
    res = {}
    for f in sorted(glob.glob(os.path.join(a.out_dir, "internals_*.json"))):
        r = json.load(open(f))
        res[r["task"]] = r
    fig_gates(res, os.path.join(a.figdir, "pooling_gates.png"))
    fig_probes(res, os.path.join(a.figdir, "layer_probes.png"))
    print(f"wrote {a.figdir}/pooling_gates.png, {a.figdir}/layer_probes.png")


if __name__ == "__main__":
    main()
