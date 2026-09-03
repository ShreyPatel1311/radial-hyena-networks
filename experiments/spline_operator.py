#!/usr/bin/env python3
"""Extract the KAN read-out operator: per-branch Jacobians, nonlinearity, activations.

    python experiments/spline_operator.py --task xpdf --checkpoint checkpoints/xpdf_seed0.pt

Produces raw artifacts consumed by experiments/spline_physics.py.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena import data as D
from radial_hyena.config import DEFAULT
from radial_hyena.operator import block_operator, collect_kan_inputs
from radial_hyena.runner import build_model, load_task, make_loader


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(D.TASKS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--data-zip", default=None)
    ap.add_argument("--out-dir", default="results/spline_operator")
    ap.add_argument("--batch-size", type=int, default=DEFAULT.batch_size)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    sub, splits, order, sg, records, out_dim = load_task(a.task, a.cache_dir, a.data_zip)
    loader = make_loader(records, order, a.task, sg, a.batch_size)
    model, ck = build_model(a.task, out_dim, a.checkpoint, a.device)
    print(f"[{a.task}] out_dim={out_dim} test={len(order)} device={a.device}", flush=True)

    X, gids = collect_kan_inputs(model, loader, a.device)
    print(f"[{a.task}] captured KAN inputs: "
          f"{ {i: tuple(v.shape) for i, v in X.items()} }", flush=True)

    out = {"task": a.task, "out_dim": out_dim, "graph_ids": gids,
           "n_test_graphs": len(order)}
    for i, b in enumerate(model.kan.blocks):
        r = block_operator(b, X[i], a.device, want_curves=(i == 0))
        for k, v in r.items():
            if v is not None:
                out[f"b{i}_{k}"] = v
        out[f"b{i}_x"] = X[i].numpy().astype(np.float32)
        out[f"b{i}_shape"] = np.array([b.in_features, b.out_features,
                                       b.grid_size + b.spline_order])
        print(f"  block {i}: in={b.in_features} out={b.out_features} "
              f"|J_base|={np.abs(r['J_base']).mean():.5f} "
              f"|J_spline|={np.abs(r['J_spline']).mean():.5f} "
              f"NL median={np.median(r['NL']):.4f}", flush=True)

    p = os.path.join(a.out_dir, f"operator_{a.task}.npz")
    np.savez_compressed(p, **out)
    print(f"[{a.task}] saved -> {p} ({os.path.getsize(p) / 1e6:.1f} MB, "
          f"{time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
