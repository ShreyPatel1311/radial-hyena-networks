#!/usr/bin/env python3
"""Evaluate a checkpoint on a CHILI-100K split.

    python scripts/evaluate.py --task crystal_system --checkpoint checkpoints/crystal_system_seed0.pt

Reports the metric alongside the value stored in the checkpoint.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena import data as D
from radial_hyena.config import DEFAULT
from radial_hyena.metrics import evaluate, metric_key
from radial_hyena.runner import OOM_HELP, build_model, load_task, make_loader


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(D.TASKS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--data-zip", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--batch-size", type=int, default=DEFAULT.batch_size)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    sub, splits, order, sg, records, out_dim = load_task(a.task, a.cache_dir, a.data_zip,
                                                         a.split)
    loader = make_loader(records, order, a.task, sg, a.batch_size)
    model, ck = build_model(a.task, out_dim, a.checkpoint, a.device)
    key = metric_key(a.task)
    try:
        m = evaluate(model, loader, a.task, a.device)
    except torch.OutOfMemoryError:
        print(OOM_HELP, file=sys.stderr)
        raise SystemExit(1)

    print(f"task={a.task} split={a.split} n={len(order)} device={a.device}")
    print(json.dumps(m, indent=2))
    if a.split == "test" and "test" in ck:
        ref = ck["test"][key]
        d = abs(m[key] - ref)
        print(f"\ncheckpoint recorded {key}={ref:.6f} | measured {m[key]:.6f} | "
              f"|delta|={d:.2e} -> {'REPRODUCED' if d <= 2e-3 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
