#!/usr/bin/env python3
"""Train one Radial Hyena model (one task, one seed) on the CHILI-100K benchmark subset.

    python scripts/train.py --task crystal_system --seed 0
    python scripts/train.py --task space_group --seed 0                     # 230-class head
    python scripts/train.py --task space_group --seed 0 --sg-vocab train    # train-split classes

Protocol of the released runs: AdamW (lr 3e-4, weight decay 1e-4); ReduceLROnPlateau on
the validation metric (factor 0.5, patience 8, min lr 1e-6); early stopping after 25 epochs
without validation improvement (at most 150 epochs); batch size 16; gradient clipping at
norm 1.0; dropout 0.1 in the KAN read-out; training-time node dropping with p = 0.2.
Train-slice, validation and test metrics are logged every epoch; model selection, early
stopping and the learning-rate schedule use the validation metric only. The best-validation
checkpoint is reloaded and evaluated on the test split at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena.config import (CLASSIFICATION, MODEL, TASKS, TRAIN, TRAIN_EVAL_SEED,   # noqa: E402
                                 level_of, metric_key)
from radial_hyena.data import CHILIDataset, label_space, load_benchmark                   # noqa: E402
from radial_hyena.kan import KAN                                                          # noqa: E402
from radial_hyena.metrics import bootstrap_ci, evaluate                                  # noqa: E402
from radial_hyena.model import RadialHyena, count_parameters                              # noqa: E402


def _versions():
    import sklearn
    import torch_geometric
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "numpy": np.__version__, "sklearn": sklearn.__version__,
            "torch_geometric": torch_geometric.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sg-vocab", choices=["all", "train"], default="all")
    ap.add_argument("--data", default="data/chili100k_benchmark.h5")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=TRAIN.epochs)
    ap.add_argument("--num-workers", type=int, default=TRAIN.num_workers)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimiser steps (smoke test)")
    a = ap.parse_args()
    task, seed, dev = a.task, a.seed, a.device
    cfg, T = MODEL, TRAIN
    os.makedirs(a.out_dir, exist_ok=True)

    # ------------------------------------------------------------------ determinism
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)   # scatter_add has no deterministic CUDA kernel
    gen = torch.Generator()
    gen.manual_seed(seed)

    def seed_worker(wid):
        np.random.seed(seed + wid); random.seed(seed + wid)

    # ------------------------------------------------------------------ data
    bench = load_benchmark(a.data)
    assert bench.check_split(), "benchmark split does not match the official split"
    tr, va, te = bench.split["train"], bench.split["val"], bench.split["test"]
    out_dim, sg_to_idx = label_space(task, bench, a.sg_vocab)
    ds = CHILIDataset(bench.records, task, sg_to_idx)                            # val / test: no augmentation
    ds_aug = CHILIDataset(bench.records, task, sg_to_idx, node_drop=T.node_drop)  # training only
    kw = dict(num_workers=a.num_workers, pin_memory=dev.startswith("cuda"),
              worker_init_fn=seed_worker, generator=gen)
    tr_eval = list(np.random.default_rng(TRAIN_EVAL_SEED).choice(tr, size=min(len(va), len(tr)), replace=False))
    tel = DataLoader(Subset(ds, tr_eval), batch_size=T.batch_size, shuffle=False, **kw)
    tl = DataLoader(Subset(ds_aug, tr), batch_size=T.batch_size, shuffle=True, **kw)
    vl = DataLoader(Subset(ds, va), batch_size=T.batch_size, shuffle=False, **kw)
    el = DataLoader(Subset(ds, te), batch_size=T.batch_size, shuffle=False, **kw)
    print(f"{task} seed {seed}: train {len(tr)} / val {len(va)} / test {len(te)} graphs, out_dim {out_dim}")

    # random-number stream of the released runs: one shuffled training batch was drawn and a
    # small KAN built before the model was initialised
    next(iter(tl))
    KAN([12, 32, 7])

    # ------------------------------------------------------------------ model
    model = RadialHyena(out_dim, level=level_of(task), cfg=cfg).to(dev)
    n_params = count_parameters(model)
    print(f"parameters: {n_params:,}")
    is_cls, key = task in CLASSIFICATION, metric_key(task)
    better = (lambda x, y: x > y) if is_cls else (lambda x, y: x < y)
    crit = nn.CrossEntropyLoss() if is_cls else None
    opt = torch.optim.AdamW(model.parameters(), lr=T.lr, weight_decay=T.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max" if is_cls else "min",
                                                       factor=T.lr_factor, patience=T.patience_lr,
                                                       min_lr=T.min_lr)
    name = f"space_group_trainvocab_seed{seed}" if (task == "space_group" and a.sg_vocab == "train") \
        else f"{task}_seed{seed}"
    ckpt = os.path.join(a.out_dir, f"{name}.pt")

    # ------------------------------------------------------------------ training
    best, best_ep, best_train, hist, steps, n_oom_total = None, 0, None, [], 0, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, nb, n_oom = time.time(), 0.0, 0, 0
        for bt in tl:
            try:
                bt = bt.to(dev)
                opt.zero_grad()
                out = model(bt)
                if is_cls:
                    t = bt.y.view(-1) if task != "atom" else bt.y
                    m = t >= 0
                    if m.sum() == 0:
                        continue
                    loss = crit(out[m], t[m])
                else:
                    loss = ((out.float() - bt.y) ** 2).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), T.grad_clip)
                opt.step()
                tot += float(loss.detach()); nb += 1; steps += 1
            except torch.cuda.OutOfMemoryError:
                n_oom += 1                           # counted and reported; the batch is skipped
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            if a.max_steps and steps >= a.max_steps:
                break
        trm, vm, tm = evaluate(model, tel, task, dev), evaluate(model, vl, task, dev), evaluate(model, el, task, dev)
        sched.step(vm[key])
        lr = opt.param_groups[0]["lr"]
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "train": trm, "val": vm, "test": tm,
                     "lr": lr, "batches_ok": nb, "batches_oom_skipped": n_oom})
        n_oom_total += n_oom
        star = ""
        if best is None or better(vm[key], best[key]):
            best, best_ep, best_train = vm, ep, trm
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": ep, "val": vm, "test": tm, "task": task, "seed": seed,
                        "out_dim": out_dim, "params": n_params, "sg_vocab": a.sg_vocab,
                        "config": {**cfg.to_dict(), **T.to_dict()}}, ckpt)
            star = "  <- best (saved)"
        print(f"ep {ep:3d} loss {tot / max(nb, 1):.5f} | train {trm[key]:.5g} | val {vm[key]:.5g} | "
              f"test {tm[key]:.5g} | lr {lr:.1e} | {time.time() - t0:.0f}s"
              + (f" | oom-skipped {n_oom}" if n_oom else "") + star, flush=True)
        if a.max_steps and steps >= a.max_steps:
            break
        if ep - best_ep >= T.patience_es:
            print(f"early stop at epoch {ep}")
            break

    # ------------------------------------------------------------------ final test
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    final_test, y, p = evaluate(model, el, task, dev, return_raw=True)
    final_val = evaluate(model, vl, task, dev)
    assert abs(final_val[key] - best[key]) < 1e-6, "reloaded checkpoint does not reproduce best validation"
    res = {"task": task, "seed": seed, "params": n_params, "out_dim": out_dim, "sg_vocab": a.sg_vocab,
           "best_epoch": best_ep, "best_val": best, "train_at_best_val": best_train,
           "test_at_best_val": final_test, "test_bootstrap_ci": bootstrap_ci(task, y, p),
           "config": {**cfg.to_dict(), **T.to_dict()},
           "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
           "split_indices": {"train": tr, "val": va, "test": te},
           "batches_completed": steps, "batches_oom_skipped": n_oom_total,
           "provenance": _versions(), "history": hist}
    with open(os.path.join(a.out_dir, f"{name}_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nbest epoch {best_ep}: val {best}\ntest {final_test}")


if __name__ == "__main__":
    main()
