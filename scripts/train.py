#!/usr/bin/env python3
"""Train a Radial Hyena network on one CHILI-100K task.

    python scripts/train.py --task crystal_system --seed 0

Model selection and the LR schedule use the validation split only. Each checkpoint stores
the weights and the val/test metrics from the same epoch.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena import data as D
from radial_hyena.config import DEFAULT
from radial_hyena.metrics import evaluate, metric_key, is_better, CLASSIFICATION_TASKS
from radial_hyena.model import RadialHyenaNet


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"  [warn] deterministic algorithms unavailable: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(D.TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-zip", default=None)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=DEFAULT.epochs)
    ap.add_argument("--batch-size", type=int, default=DEFAULT.batch_size)
    ap.add_argument("--lr", type=float, default=DEFAULT.lr)
    ap.add_argument("--weight-decay", type=float, default=DEFAULT.weight_decay)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fix-ordering", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    dev = a.device

    index = D.build_index(a.data_zip, cache=os.path.join(a.cache_dir, "chili_index.pkl"))
    sub = D.benchmark_subset(index)
    tr, va, te = D.make_split(sub)
    sg_to_idx = D.space_group_vocab(sub, tr) if a.task == "space_group" else None
    records = D.load_records(sub, sorted(tr + va + te), a.data_zip,
                             cache=os.path.join(a.cache_dir, "records.pt"))
    out_dim = D.out_dim_for(a.task, sub, tr, records)
    level = "node" if a.task == "atom" else "graph"
    key = metric_key(a.task)

    mk = lambda order, aug=False: DataLoader(
        D.CHILIDataset(records, order, a.task, sg_to_idx=sg_to_idx, augment=aug),
        batch_size=a.batch_size, shuffle=aug, num_workers=2)
    tl, vl, el = mk(tr, True), mk(va), mk(te)
    tr_eval = list(np.random.default_rng(12345).choice(tr, size=len(va), replace=False))
    tel = mk(tr_eval)

    model = RadialHyenaNet(out_dim, level=level, fix_ordering=a.fix_ordering).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"task={a.task} seed={a.seed} level={level} out_dim={out_dim} "
          f"params={n_params:,} device={dev}")

    crit = nn.CrossEntropyLoss() if a.task in CLASSIFICATION_TASKS else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max" if a.task in CLASSIFICATION_TASKS else "min",
        factor=0.5, patience=DEFAULT.patience_lr, min_lr=1e-6)

    best = best_test = best_ep = None
    history, ckpt = [], os.path.join(a.out_dir, f"{a.task}_seed{a.seed}.pt")
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = nb = n_oom = 0
        for bt in tl:
            try:
                bt = bt.to(dev); opt.zero_grad()
                out = model(bt)
                if a.task in CLASSIFICATION_TASKS:
                    t = bt.y.view(-1) if a.task != "atom" else bt.y
                    m = t >= 0
                    if m.sum() == 0:
                        continue
                    loss = crit(out[m], t[m])
                else:
                    loss = ((out.float() - bt.y) ** 2).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()); nb += 1
            except torch.cuda.OutOfMemoryError:
                n_oom += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        trm = evaluate(model, tel, a.task, dev)
        vm = evaluate(model, vl, a.task, dev)
        tm = evaluate(model, el, a.task, dev)
        sched.step(vm[key])
        history.append({"epoch": ep, "train_loss": tot / max(nb, 1), "train": trm,
                        "val": vm, "test": tm, "lr": opt.param_groups[0]["lr"],
                        "batches_ok": nb, "batches_oom_skipped": n_oom})
        star = ""
        if best is None or is_better(a.task, vm[key], best[key]):
            best, best_test, best_ep = vm, tm, ep
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": ep, "val": vm, "test": tm,
                        "task": a.task, "seed": a.seed, "out_dim": out_dim,
                        "params": n_params, "config": vars(DEFAULT),
                        "fix_ordering": a.fix_ordering}, ckpt)
            star = "  <-- best (saved)"
        print(f"ep {ep:3d} loss {tot / max(nb, 1):.4f} | train {trm[key]:.4f} "
              f"val {vm[key]:.4f} test {tm[key]:.4f} | {time.time() - t0:.0f}s"
              + (f" | {n_oom} OOM skipped" if n_oom else "") + star, flush=True)
        if ep - best_ep >= DEFAULT.patience_es:
            print(f"early stop: no val improvement in {DEFAULT.patience_es} epochs")
            break

    json.dump({"task": a.task, "seed": a.seed, "best_epoch": best_ep, "val": best,
               "test": best_test, "params": n_params, "history": history},
              open(os.path.join(a.out_dir, f"{a.task}_seed{a.seed}_history.json"), "w"),
              indent=2)
    print(f"\nbest epoch {best_ep}: val {best[key]:.4f} | test {best_test[key]:.4f}")
    print(f"checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
