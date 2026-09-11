#!/usr/bin/env python3
"""Stream ablation: contribution of each stream to each trained model (seed 0, test split).

The KAN read-out consumes [node (64) | edge (64) | graph (64)]. A stream is removed by
zeroing its slice of that vector before the KAN; no weight changes. The KAN input is
computed once per batch and read out for the unablated model and for each ablation.
Degradation is reported relative to the unablated model (classification: drop in weighted
F1; regression: increase in MSE), in percent.

Outputs: results/analysis/stream_ablation_<task>.json and figures/stream_ablation.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DEFAULT_DEVICE, TASKS, load, run                                  # noqa: E402
from radial_hyena.config import CLASSIFICATION, MODEL, NICE, metric_key               # noqa: E402
from radial_hyena.data import load_benchmark                                          # noqa: E402
from radial_hyena.metrics import classification_metrics, targets                      # noqa: E402

SLICES = {"node": (0, MODEL.node_dim),
          "edge": (MODEL.node_dim, MODEL.node_dim + MODEL.edge_dim),
          "graph": (MODEL.node_dim + MODEL.edge_dim, MODEL.node_dim + MODEL.edge_dim + MODEL.graph_dim)}
VARIANTS = {"baseline": None, **{f"zero_{k}": v for k, v in SLICES.items()}}
COL = {"node": "#1f77b4", "edge": "#d62728", "graph": "#7f7f7f"}
LBL = {"node": "node stream (global)", "edge": "edge stream (local)", "graph": "graph stream"}


def _variants(m, bt):
    h_v, h_e, g, batch, B, _ = m.trunk(bt)
    x = m.readout_input(h_v, h_e, g, batch, B, bt.edge_index)
    out = {}
    for name, sl in VARIANTS.items():
        xv = x
        if sl is not None:
            xv = x.clone()
            xv[..., sl[0]:sl[1]] = 0.0
        out[name] = m.kan(xv).float()
    return out


@torch.no_grad()
def ablate(model, cpu_model, loader, task, device):
    model.eval(); cpu_model.eval()
    ys, ps = [], {k: [] for k in VARIANTS}
    se, n = {k: 0.0 for k in VARIANTS}, 0
    for bt in loader:
        y = targets(bt, task).cpu()
        outs = run(_variants, model, cpu_model, bt, device)
        if task in CLASSIFICATION:
            ys.append(y)
            for k, o in outs.items():
                ps[k].append(o.argmax(-1))
        else:
            n += y.numel()
            for k, o in outs.items():
                se[k] += ((o - y) ** 2).sum().item()
    if task in CLASSIFICATION:
        y = torch.cat(ys).numpy()
        return {k: classification_metrics(y, torch.cat(ps[k]).numpy())[0] for k in VARIANTS}
    return {k: {"mse": se[k] / n, "n_eval": int(n)} for k in VARIANTS}


def degradation(r, stream):
    k = r["metric_key"]
    b, v = r["runs"]["baseline"][k], r["runs"][f"zero_{stream}"][k]
    sign = -1 if k == "weighted_f1" else 1
    return 100 * sign * (v - b) / abs(b)


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = [t for t in TASKS if t in res]
    x, w = np.arange(len(tasks)), 0.27
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 8.6))
    for i, s in enumerate(["node", "edge", "graph"]):
        v = np.array([max(degradation(res[t], s), 1e-2) for t in tasks])
        axs[0].bar(x + (i - 1) * w, v, w, color=COL[s], label=LBL[s])
    axs[0].set_yscale("log")
    axs[0].axhline(100, color="k", lw=.8, ls=":")
    axs[0].set_ylabel("degradation when stream is removed (%)")
    axs[0].legend(fontsize=8.5)
    for i, s in enumerate(["node", "edge", "graph"]):
        frac = []
        for t in tasks:
            d = {q: max(degradation(res[t], q), 0.0) for q in ("node", "edge", "graph")}
            frac.append(100 * d[s] / (sum(d.values()) or 1.0))
        axs[1].bar(x + (i - 1) * w, frac, w, color=COL[s], label=LBL[s])
    axs[1].axhline(50, color="k", lw=.8, ls=":")
    axs[1].set_ylabel("share of total degradation (%)")
    axs[1].set_ylim(0, 100)
    for i, ax in enumerate(axs):
        ax.set_xticks(x)
        ax.set_xticklabels([NICE[t] for t in tasks], rotation=20, ha="right", fontsize=9)
        ax.grid(alpha=.25, axis="y")
        ax.text(-0.10, 1.02, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="bottom", ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--data", default="data/chili100k_benchmark.h5")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--sg-vocab", choices=["all", "train"], default="train")
    ap.add_argument("--out-dir", default=os.path.join("results", "analysis"))
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.figdir, exist_ok=True)

    if not a.plot_only:
        bench = load_benchmark(a.data)
        for task in a.tasks:
            model, cpu, meta, loader = load(task, bench, a.ckpt_dir, a.device, sg_vocab=a.sg_vocab)
            key = metric_key(task)
            runs = ablate(model, cpu, loader, task, a.device)
            r = {"task": task, "metric_key": key, "recorded_test": meta.get("test"),
                 "slices": SLICES, "runs": runs}
            with open(os.path.join(a.out_dir, f"stream_ablation_{task}.json"), "w") as f:
                json.dump(r, f, indent=2)
            print(f"{task:15s} baseline {runs['baseline'][key]:.5g} | "
                  + " | ".join(f"no {s}: {degradation(r, s):+.1f}%" for s in SLICES), flush=True)
            del model, cpu
            torch.cuda.empty_cache() if a.device.startswith("cuda") else None
    res = {}
    for f in sorted(glob.glob(os.path.join(a.out_dir, "stream_ablation_*.json"))):
        r = json.load(open(f))
        res[r["task"]] = r
    figure(res, os.path.join(a.figdir, "stream_ablation.png"))
    rows = ["| Task | Metric | Baseline | Node stream removed | Edge stream removed | Graph stream removed |",
            "|---|---|---|---|---|---|"]
    for t in [t for t in TASKS if t in res]:
        r, k = res[t], res[t]["metric_key"]
        f = (lambda v: f"{v:.4f}") if t in CLASSIFICATION else (lambda v: f"{v:.5f}")
        rows.append(f"| {NICE[t]} | {k} | {f(r['runs']['baseline'][k])} | "
                    + " | ".join(f"{f(r['runs'][f'zero_{s}'][k])} ({degradation(r, s):+.1f}%)"
                                 for s in SLICES) + " |")
    with open(os.path.join(a.out_dir, "stream_ablation.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
