"""CHILI-100K data pipeline.

Follows the official CHILI benchmark protocol:

  1. index every (material, particle-size) pair in the dataset archive;
  2. take a stratified subset of `subset_per_class` graphs per crystal system
     (425 -> 2,975 graphs) using numpy's default_rng(42);
  3. split 80/10/10 with sklearn train_test_split, stratified by space-group number,
     random_state=42, over individual graphs.

This yields train 2,379 / val 298 / test 298, with 151 train-only space-group classes and
3 test graphs outside that vocabulary.

The archive is read lazily: .h5 members are pulled from the zip into memory one at a time
rather than extracting 14.5 GB to disk.
"""
from __future__ import annotations
import io
import os
import pickle
import re
import zipfile

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .config import ModelConfig, DEFAULT
from .model import rbf_encode

CRYSTAL_SYSTEMS = {"Triclinic": 0, "Monoclinic": 1, "Orthorhombic": 2, "Tetragonal": 3,
                   "Trigonal": 4, "Hexagonal": 5, "Cubic": 6}
SCATTER_FIELD = {"saxs": "SAXS", "xrd": "XRD", "xpdf": "xPDF"}
TASKS = ("crystal_system", "space_group", "atom", "saxs", "xrd", "xpdf")


def build_index(zip_path, cache=None, verbose=True):
    """One entry per (material file, particle size): (path, key, crystal_system, sg, rank)."""
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    import h5py
    zf = zipfile.ZipFile(zip_path)
    members = sorted(n for n in zf.namelist() if n.endswith(".h5"))
    index = []
    for j, fp in enumerate(members):
        try:
            with h5py.File(io.BytesIO(zf.read(fp)), "r") as f:
                g = f["GlobalLabels"]
                cs = g["CrystalSystem"][()]
                cs = cs.decode() if isinstance(cs, (bytes, bytearray)) else str(cs)
                cs = CRYSTAL_SYSTEMS.get(cs, -1)
                sg = int(g["SpaceGroupNumber"][()])
                keys = sorted(f["DiscreteParticleGraphs"].keys(),
                              key=lambda k: float(re.findall(r"[\d.]+", k)[0]))
                for rank, k in enumerate(keys):
                    index.append((fp, k, cs, sg, rank))
        except Exception:
            continue
        if verbose and (j + 1) % 2000 == 0:
            print(f"  indexed {j + 1}/{len(members)} files", flush=True)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        pickle.dump(index, open(cache, "wb"))
    return index


def benchmark_subset(index, subset_per_class=DEFAULT.subset_per_class):
    """Stratified subset of `subset_per_class` graphs per crystal system."""
    if not subset_per_class:
        return list(index)
    rng = np.random.default_rng(42)
    by_cs = {}
    for i, rec in enumerate(index):
        if rec[2] >= 0:
            by_cs.setdefault(rec[2], []).append(i)
    picked = []
    for c in sorted(by_cs):
        pool = np.array(by_cs[c])
        picked += rng.choice(pool, size=min(subset_per_class, len(pool)),
                             replace=False).tolist()
    return [index[i] for i in sorted(picked)]


def make_split(index):
    """80/10/10 over individual graphs, stratified by space group."""
    idx = np.arange(len(index))
    strat = np.array([r[3] for r in index])
    _, counts = np.unique(strat, return_counts=True)
    if (counts < 2).any():
        strat = None
    tr, te = train_test_split(idx, test_size=0.10, random_state=42, stratify=strat)
    strat_tr = strat[tr] if strat is not None else None
    tr, va = train_test_split(tr, test_size=0.10 / 0.90, random_state=42, stratify=strat_tr)
    return tr.tolist(), va.tolist(), te.tolist()


def space_group_vocab(index, train_idx):
    """Train-only space-group vocabulary."""
    classes = sorted(set(int(index[i][3]) for i in train_idx))
    return {s: i for i, s in enumerate(classes)}


def out_dim_for(task, index, train_idx, records=None):
    if task == "crystal_system":
        return 7
    if task == "atom":
        return 118
    if task == "space_group":
        return len(space_group_vocab(index, train_idx))
    if records is None:
        raise ValueError(f"records needed to size the {task} target")
    return len(next(iter(records.values()))["scatter"][task])


def load_records(index, which, zip_path, cache=None, all_scatter=True, verbose=True):
    """Read the h5 payload for the graphs at positions `which` (into `index`)."""
    if cache and os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    import h5py
    zf = zipfile.ZipFile(zip_path)
    out = {}
    for n, i in enumerate(which):
        fp, k, cs, sg, rank = index[i]
        with h5py.File(io.BytesIO(zf.read(fp)), "r") as f:
            d = f["DiscreteParticleGraphs"][k]
            rec = dict(nf=d["NodeFeatures"][:].astype(np.float32),
                       ei=d["EdgeDirections"][:].astype(np.int64),
                       pos=d["AbsoluteCoordinates"][:].astype(np.float32),
                       cs=cs, sg=sg, rank=rank)
            if all_scatter:
                rec["scatter"] = {t: f["ScatteringData"][k][fld][1].astype(np.float32)
                                  for t, fld in SCATTER_FIELD.items()}
        out[i] = rec
        if verbose and (n + 1) % 250 == 0:
            print(f"  loaded {n + 1}/{len(which)} graphs", flush=True)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        torch.save(out, cache)
    return out


def scattering_axis(zip_path, index, field="xPDF"):
    """The physical axis: q for SAXS/XRD, r for xPDF."""
    import h5py
    zf = zipfile.ZipFile(zip_path)
    fp, k = index[0][0], index[0][1]
    with h5py.File(io.BytesIO(zf.read(fp)), "r") as f:
        return f["ScatteringData"][k][field][0][:].astype(np.float64)


class CHILIDataset(Dataset):
    """Graphs for one task."""

    def __init__(self, records, order, task, cfg: ModelConfig = DEFAULT,
                 sg_to_idx=None, augment=False, pos_jitter=0.0, node_drop=0.0):
        self.records, self.order, self.task, self.cfg = records, order, task, cfg
        self.sg_to_idx = sg_to_idx
        self.augment, self.pos_jitter, self.node_drop = augment, pos_jitter, node_drop

    def __len__(self):
        return len(self.order)

    def __getitem__(self, j):
        cfg, task = self.cfg, self.task
        r = self.records[self.order[j]]
        nf = torch.from_numpy(r["nf"])
        pos = torch.from_numpy(r["pos"])
        ei = torch.from_numpy(r["ei"]).long()

        if self.augment and self.node_drop > 0 and nf.shape[0] > 24:
            keep = torch.rand(nf.shape[0]) > self.node_drop
            if int(keep.sum()) >= 12:
                remap = torch.full((nf.shape[0],), -1, dtype=torch.long)
                remap[keep] = torch.arange(int(keep.sum()))
                emask = keep[ei[0]] & keep[ei[1]]
                ei = remap[ei[:, emask]]
                nf, pos = nf[keep], pos[keep]
        if self.augment and self.pos_jitter > 0:
            pos = pos + torch.randn_like(pos) * self.pos_jitter

        posc = pos - pos.mean(0, keepdim=True)
        if ei.numel() == 0:
            ei = torch.zeros(2, 1, dtype=torch.long)
        vec = posc[ei[1]] - posc[ei[0]]
        rd = vec.norm(dim=-1).clamp(min=1e-6)

        if task == "atom":
            z = torch.zeros(nf.shape[0], dtype=torch.long)
            cont = posc / cfg.pos_scale
        else:
            z = nf[:, 0].long().clamp(0, 118)
            cont = nf[:, 1:4]

        ga = torch.tensor([np.log1p(nf.shape[0]), float(len(torch.unique(nf[:, 0]))),
                           float(r["rank"])], dtype=torch.float32).unsqueeze(0)

        d = Data(x=cont, z=z, edge_index=ei,
                 edge_attr=rbf_encode(rd, cfg.num_rbf_pos, 0.0, cfg.rbf_cutoff),
                 edge_unit_vec=vec / rd.unsqueeze(-1),
                 edge_raw_dist=rd, graph_attr=ga, pos=posc)
        if task == "crystal_system":
            d.y = torch.tensor([max(r["cs"], 0)], dtype=torch.long)
        elif task == "space_group":
            d.y = torch.tensor([self.sg_to_idx.get(int(r["sg"]), -1)], dtype=torch.long)
        elif task == "atom":
            d.y = nf[:, 0].long().clamp(1, 118) - 1
        else:
            it = torch.from_numpy(r["scatter"][task])
            # per-sample min-max normalisation, matching the CHILI baseline protocol
            d.y = ((it - it.min()) / (it.max() - it.min() + 1e-9)).unsqueeze(0)
        return d
