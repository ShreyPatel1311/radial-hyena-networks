#!/usr/bin/env python3
"""Evaluate released checkpoints on the benchmark test split and rebuild the results table.

    python scripts/evaluate.py                          # all tasks, seeds 0-2
    python scripts/evaluate.py --tasks saxs --seeds 0

Each run is compared with the test metric recorded in its checkpoint at training time.
The table reports mean ± population standard deviation over seeds.
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
from radial_hyena.checkpoints import restore, test_loader                    # noqa: E402
from radial_hyena.config import NICE, TASKS, metric_key                     # noqa: E402
from radial_hyena.data import load_benchmark                                 # noqa: E402
from radial_hyena.metrics import bootstrap_ci, evaluate                     # noqa: E402


def ckpt_name(task, seed, sg_vocab="all"):
    if task == "space_group" and sg_vocab == "train":
        return f"space_group_trainvocab_seed{seed}.pt"
    return f"{task}_seed{seed}.pt"


def fmt(task, v):
    return f"{v:.4f}" if metric_key(task) != "mse" else f"{v:.5f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--sg-vocab", choices=["all", "train"], default="all",
                    help="space-group head: all 230 groups (reported) or the train-split vocabulary")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--data", default="data/chili100k_benchmark.h5")
    ap.add_argument("--out-dir", default="results/evaluation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    bench = load_benchmark(a.data)
    assert bench.check_split(), "benchmark split does not match the official split"
    rows, runs = [], {}
    for task in a.tasks:
        key = metric_key(task)
        vals = []
        for seed in a.seeds:
            path = os.path.join(a.ckpt_dir, ckpt_name(task, seed, a.sg_vocab))
            if not os.path.exists(path):
                print(f"[skip] {path} not found")
                continue
            t0 = time.time()
            model, meta, cpu = restore(path, a.device, with_cpu_copy=True)
            vocab = meta.get("sg_vocab", "all")
            loader = test_loader(bench, task, vocab)
            m, y, p = evaluate(model, loader, task, a.device, cpu, return_raw=True)
            m["bootstrap_ci"] = bootstrap_ci(task, y, p)
            rec = meta.get("test", {}).get(key)
            diff = None if rec is None else abs(m[key] - rec)
            print(f"{task:15s} seed {seed}: {key} = {m[key]:.6g}"
                  + ("" if rec is None else f"   (recorded {rec:.6g}, |diff| = {diff:.1e})")
                  + f"   [{time.time() - t0:.0f}s]", flush=True)
            runs[f"{task}_seed{seed}"] = {"task": task, "seed": seed, "checkpoint": os.path.basename(path),
                                          "out_dim": meta["out_dim"], "params": meta.get("params"),
                                          "test": m, "recorded_test": meta.get("test"),
                                          "abs_diff_vs_recorded": diff}
            vals.append(m[key])
            del model, cpu
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
        if vals:
            v = np.array(vals)
            rows.append((task, key, v.mean(), v.std(), len(v)))

    lines = ["| Task | Metric | Radial Hyena (mean ± std) | seeds |", "|---|---|---|---|"]
    for task, key, mu, sd, n in rows:
        name = "weighted F1 ↑" if key != "mse" else "MSE ↓"
        lines.append(f"| {NICE[task]} | {name} | {fmt(task, mu)} ± {fmt(task, sd)} | {n} |")
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(a.out_dir, "benchmark_table.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(a.out_dir, "evaluation.json"), "w") as f:
        json.dump({"device": a.device, "torch": torch.__version__, "runs": runs,
                   "table": [dict(task=t, metric=k, mean=mu, std=sd, n_seeds=n) for t, k, mu, sd, n in rows]},
                  f, indent=2)


if __name__ == "__main__":
    main()
