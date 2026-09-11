#!/usr/bin/env python3
"""Dataset figures for the CHILI-100K benchmark subset.

  data_crystal_systems.png   crystal-system shares, full dataset vs 425-per-class subset
  data_space_groups.png      space-group frequencies in the subset (train coverage marked)
  data_normalisation.png     SAXS / XRD / xPDF curves before and after per-sample
                             min-max normalisation (six random graphs)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402,F401  (repository root on sys.path)
from radial_hyena.config import CRYSTAL_SYSTEMS                   # noqa: E402
from radial_hyena.data import load_benchmark                      # noqa: E402

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

TARGETS = [("saxs", "SAXS", "q  (Å$^{-1}$)", "intensity (a.u.)"),
           ("xrd", "XRD", "q  (Å$^{-1}$)", "intensity (a.u.)"),
           ("xpdf", "xPDF", "r  (Å)", "G(r)")]


def _sublabel(ax, i, y=1.03):
    ax.text(-0.08, y, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")


def fig_crystal_systems(bench, path):
    full = bench.full_crystal_system_counts
    sub = Counter(e[2] for e in bench.entries)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    cols = plt.cm.tab10.colors
    for ax, v, ttl in [(axs[0], [full[s] for s in CRYSTAL_SYSTEMS], "Full dataset"),
                       (axs[1], [sub.get(c, 0) for c in range(7)], "Benchmark subset")]:
        ax.pie(v, labels=CRYSTAL_SYSTEMS, autopct=lambda p: f"{p:.1f}%", startangle=90,
               colors=cols[:7], textprops={"fontsize": 8.5},
               wedgeprops={"edgecolor": "w", "linewidth": .6})
        ax.set_title(f"{ttl}  (n = {sum(v):,})", fontsize=10)
    for i, ax in enumerate(axs):
        _sublabel(ax, i)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig_space_groups(bench, path):
    cnt = Counter(e[3] for e in bench.entries)
    sg_tr = set(bench.entries[i][3] for i in bench.split["train"])
    order = [s for s, _ in cnt.most_common()]
    vals = [cnt[s] for s in order]
    fig, axs = plt.subplots(2, 1, figsize=(11, 8))
    axs[0].bar(range(len(order)), vals, width=1.0,
               color=["#1f77b4" if s in sg_tr else "#d62728" for s in order])
    axs[0].set_yscale("log")
    axs[0].set_xlabel(f"space group, ranked by frequency ({len(order)} present of 230)")
    axs[0].set_ylabel("graphs in subset")
    axs[0].plot([], [], color="#1f77b4", label="present in train")
    axs[0].plot([], [], color="#d62728", label="absent from train")
    axs[0].legend(fontsize=8.5); axs[0].grid(alpha=.25, axis="y")
    top = 25
    axs[1].bar(range(top), vals[:top], color="#1f77b4")
    axs[1].set_xticks(range(top))
    axs[1].set_xticklabels([str(s) for s in order[:top]], fontsize=8, rotation=60)
    axs[1].set_xlabel("space group number (25 most frequent)")
    axs[1].set_ylabel("graphs in subset"); axs[1].grid(alpha=.25, axis="y")
    for i, ax in enumerate(axs):
        _sublabel(ax, i)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig_normalisation(bench, path, n_show=6):
    pick = list(np.random.default_rng(0).choice(len(bench), size=n_show, replace=False))
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.4))
    cmap = plt.cm.viridis(np.linspace(0, .85, n_show))
    for col, (t, nice, xlab, ylab) in enumerate(TARGETS):
        ax_r, ax_n = axs[0][col], axs[1][col]
        x = bench.axes[t]
        for c, k in zip(cmap, pick):
            y = bench.records[k]["scatter"][t].astype(np.float64)
            ax_r.plot(x, y, lw=.9, color=c)
            ax_n.plot(x, (y - y.min()) / (y.max() - y.min() + 1e-9), lw=.9, color=c)
        if t != "xpdf":
            ax_r.set_yscale("log")
        ax_r.set_title(nice, fontsize=10)
        if col == 0:
            ax_r.set_ylabel(ylab + ("  (log)" if t != "xpdf" else ""))
            ax_n.set_ylabel("normalised to [0, 1]")
        for ax in (ax_r, ax_n):
            ax.set_xlabel(xlab); ax.grid(alpha=.25)
        ax_n.set_ylim(-0.05, 1.05)
    for i, ax in enumerate(axs.ravel()):
        _sublabel(ax, i)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/chili100k_benchmark.h5")
    ap.add_argument("--figdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    bench = load_benchmark(a.data)
    fig_crystal_systems(bench, os.path.join(a.figdir, "data_crystal_systems.png"))
    fig_space_groups(bench, os.path.join(a.figdir, "data_space_groups.png"))
    fig_normalisation(bench, os.path.join(a.figdir, "data_normalisation.png"))
    cnt = Counter(e[3] for e in bench.entries)
    tr_sg = set(bench.entries[i][3] for i in bench.split["train"])
    unseen = sum(1 for i in bench.split["test"] if bench.entries[i][3] not in tr_sg)
    print(f"space groups in subset: {len(cnt)}, in train: {len(tr_sg)}, "
          f"test graphs whose group is absent from train: {unseen}")
    print(f"wrote {a.figdir}/data_crystal_systems.png, data_space_groups.png, data_normalisation.png")


if __name__ == "__main__":
    main()
