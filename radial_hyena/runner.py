"""Shared setup: load a task, build its model, check a checkpoint reproduces."""
from __future__ import annotations
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from torch_geometric.loader import DataLoader

from . import data as D
from .config import DEFAULT
from .metrics import metric_key
from .model import RadialHyenaNet

OOM_HELP = """CUDA out of memory. Roughly 8 GB is needed at the default batch size of 16.
Use a larger GPU, or --device cpu.
"""


def load_task(task, cache_dir="cache", data_zip=None, split="test"):
    """Returns (subset_index, (train, val, test), order, sg_to_idx, records, out_dim)."""
    index = D.build_index(data_zip, cache=os.path.join(cache_dir, "chili_index.pkl"))
    sub = D.benchmark_subset(index)
    tr, va, te = D.make_split(sub)
    order = {"train": tr, "val": va, "test": te, "all": sorted(tr + va + te)}[split]
    sg_to_idx = D.space_group_vocab(sub, tr) if task == "space_group" else None
    records = D.load_records(sub, sorted(tr + va + te), data_zip,
                             cache=os.path.join(cache_dir, "records.pt"))
    out_dim = D.out_dim_for(task, sub, tr, records)
    return sub, (tr, va, te), order, sg_to_idx, records, out_dim


def make_loader(records, order, task, sg_to_idx=None, batch_size=DEFAULT.batch_size):
    return DataLoader(D.CHILIDataset(records, order, task, sg_to_idx=sg_to_idx),
                      batch_size=batch_size, shuffle=False)


def build_model(task, out_dim, checkpoint, device, fix_ordering=False):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert ck["task"] == task, f"checkpoint is for {ck['task']}, not {task}"
    assert ck["out_dim"] == out_dim, "checkpoint/task output size mismatch"
    model = RadialHyenaNet(out_dim, level="node" if task == "atom" else "graph",
                           fix_ordering=fix_ordering).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    if "params" in ck:
        assert n == ck["params"], f"param count {n} != checkpoint {ck['params']}"
    return model, ck


def reproduction_gate(measured, ck, task, tol=2e-3, label=""):
    """Compare a freshly measured metric against the value stored in the checkpoint."""
    key = metric_key(task)
    ref = ck["test"][key]
    delta = abs(measured[key] - ref)
    ok = delta <= tol
    print(f"[{task}] {label}baseline {key}={measured[key]:.6f} vs checkpoint {ref:.6f} "
          f"| |delta|={delta:.2e} -> {'REPRODUCED' if ok else 'MISMATCH'}", flush=True)
    return ok
