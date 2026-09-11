"""Checkpoint loading, model construction and the released-weights format."""
from __future__ import annotations

import copy
import os

import torch
from torch_geometric.loader import DataLoader

from .config import MODEL, TRAIN, level_of
from .data import CHILIDataset, label_space, sg_vocab_of_checkpoint
from .model import RadialHyena, count_parameters

META_KEYS = ("task", "seed", "epoch", "out_dim", "params", "val", "test", "sg_vocab")


def load_checkpoint(path):
    """Returns {"model": state_dict, <metadata>}. Accepts the released model-only files and
    full training checkpoints (which additionally carry optimizer/scheduler state)."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    if "sg_vocab" not in ck and ck.get("task") == "space_group":
        ck["sg_vocab"] = sg_vocab_of_checkpoint(ck["out_dim"])
    return ck


def build_model(task, out_dim, device="cpu", state_dict=None):
    model = RadialHyena(out_dim, level=level_of(task), cfg=MODEL)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def restore(path, device="cpu", with_cpu_copy=False):
    """Checkpoint -> (model on device, metadata, optional CPU copy for out-of-memory batches)."""
    ck = load_checkpoint(path)
    model = build_model(ck["task"], ck["out_dim"], device, ck["model"])
    params = count_parameters(model)
    if "params" in ck and ck["params"] != params:
        raise ValueError(f"{path}: {params} parameters, checkpoint records {ck['params']}")
    model.eval()
    cpu = None
    if with_cpu_copy and str(device) != "cpu":
        cpu = copy.deepcopy(model).to("cpu").eval()
    meta = {k: ck[k] for k in META_KEYS if k in ck}
    return model, meta, cpu


def test_loader(bench, task, sg_vocab="all", split="test", batch_size=TRAIN.batch_size):
    """Evaluation loader: the split's graphs in their stored order, batch size 16."""
    _, sg_to_idx = label_space(task, bench, sg_vocab)
    recs = [bench.records[i] for i in bench.split[split]]
    return DataLoader(CHILIDataset(recs, task, sg_to_idx), batch_size=batch_size, shuffle=False)


def export_release(src, dst, extra=None):
    """Write a model-only copy of a training checkpoint (optimizer state dropped)."""
    ck = load_checkpoint(src)
    out = {"model": ck["model"]}
    for k in META_KEYS:
        if k in ck:
            out[k] = ck[k]
    out.update(extra or {})
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save(out, dst)
    return out
