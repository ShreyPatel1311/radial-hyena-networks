#!/usr/bin/env python3
"""KAN spline pruning curves.

    python experiments/kan_pruning.py --task crystal_system --checkpoint checkpoints/...

Every KANLinear edge computes  w_base*SiLU(x) + phi(x).  Pruning zeroes the spline branch
only, leaving the base SiLU path intact.

Axis 1: remove a fraction of edges' splines, lowest importance first, under three
  orderings (weight magnitude, activation magnitude, random control). Selection is exact
  top-k.
Axis 2: keep every edge but re-express each spline on a coarser B-spline grid.

Reference points: all splines off (base path only) and base off (splines only).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena import data as D
from radial_hyena.config import DEFAULT
from radial_hyena.metrics import evaluate, metric_key
from radial_hyena.operator import (apply_spline_mask, coarsen_spline_grid, kan_edge_count,
                                   restore_spline_grid, restore_splines, spline_importances)
from radial_hyena.runner import build_model, load_task, make_loader, reproduction_gate

FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(D.TASKS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--data-zip", default=None)
    ap.add_argument("--out-dir", default="results/kan_pruning")
    ap.add_argument("--batch-size", type=int, default=DEFAULT.batch_size)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--skip-grid", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t_start = time.time()

    sub, splits, order, sg, records, out_dim = load_task(a.task, a.cache_dir, a.data_zip)
    loader = make_loader(records, order, a.task, sg, a.batch_size)
    model, ck = build_model(a.task, out_dim, a.checkpoint, a.device)
    key = metric_key(a.task)
    print(f"[{a.task}] test graphs={len(order)} out_dim={out_dim} device={a.device}")

    base = evaluate(model, loader, a.task, a.device)
    reproduced = reproduction_gate(base, ck, a.task)

    orig_spline = [b.spline_weight.data.clone() for b in model.kan.blocks]
    orig_base = [b.base_weight.data.clone() for b in model.kan.blocks]
    orig_grid = [b.grid.clone() for b in model.kan.blocks]
    orig_gs = [b.grid_size for b in model.kan.blocks]
    n_edges = kan_edge_count(model)

    imps = spline_importances(model, loader, a.device)
    print(f"[{a.task}] {n_edges} KAN spline edges", flush=True)

    def sweep(name, ordering):
        out = {}
        for f in FRACTIONS:
            k = int(round(f * n_edges))
            keep = np.ones(n_edges, dtype=bool)
            keep[ordering[:k]] = False
            apply_spline_mask(model, orig_spline, keep)
            m = evaluate(model, loader, a.task, a.device)
            m["n_pruned"] = k
            out[f"{f:.2f}"] = m
            print(f"    {name} {100 * f:5.1f}% ({k:6d}/{n_edges}) -> {key}={m[key]:.5f}",
                  flush=True)
        restore_splines(model, orig_spline)
        return out

    curves = {}
    for crit in ("weight_l2", "activation"):
        print(f"  [axis 1] ordering = {crit}", flush=True)
        curves[crit] = sweep(crit, np.argsort(imps[crit], kind="stable"))
    print("  [axis 1] ordering = random (control)", flush=True)
    curves["random"] = {f"seed{s}": sweep(f"random s{s}",
                                          np.random.default_rng(1000 + s).permutation(n_edges))
                        for s in range(a.random_seeds)}

    refs = {}
    apply_spline_mask(model, orig_spline, np.zeros(n_edges, dtype=bool))
    refs["spline_off"] = evaluate(model, loader, a.task, a.device)
    restore_splines(model, orig_spline)
    for b in model.kan.blocks:
        b.base_weight.data = torch.zeros_like(b.base_weight.data)
    refs["base_off"] = evaluate(model, loader, a.task, a.device)
    restore_splines(model, orig_spline, orig_base)
    print(f"  [refs] spline_off {key}={refs['spline_off'][key]:.5f} | "
          f"base_off {key}={refs['base_off'][key]:.5f}", flush=True)

    grid_curve = {}
    if not a.skip_grid:
        for g in [1, 2, 3, 4, 5]:
            coarsen_spline_grid(model, orig_spline, g)
            m = evaluate(model, loader, a.task, a.device)
            m["coeffs_per_edge"] = g + model.kan.blocks[0].spline_order
            grid_curve[f"grid{g}"] = m
            restore_spline_grid(model, orig_grid, orig_spline, orig_gs)
            print(f"  [axis 2] grid_size {g} ({g + 3} coeffs/edge) -> {key}={m[key]:.5f}",
                  flush=True)

    result = {"task": a.task, "metric_key": key, "device": a.device,
              "n_test_graphs": len(order), "batch_size": a.batch_size,
              "checkpoint": {k: ck[k] for k in ("epoch", "val", "test", "params") if k in ck},
              "baseline": base, "baseline_reproduced": bool(reproduced),
              "n_kan_edges": n_edges,
              "kan_blocks": [{"in": b.in_features, "out": b.out_features,
                              "coeffs_per_edge": b.grid_size + b.spline_order}
                             for b in model.kan.blocks],
              "axis1_edge_pruning": curves, "reference_points": refs,
              "axis2_grid_coarsening": grid_curve,
              "elapsed_sec": time.time() - t_start}
    p = os.path.join(a.out_dir, f"kan_pruning_{a.task}.json")
    json.dump(result, open(p, "w"), indent=2)
    np.savez(os.path.join(a.out_dir, f"importances_{a.task}.npz"), **imps)
    print(f"[{a.task}] saved -> {p} ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
