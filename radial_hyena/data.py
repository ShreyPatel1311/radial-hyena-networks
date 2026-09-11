"""CHILI-100K benchmark subset: selection, official split, storage and PyG dataset.

The benchmark is the stratified subset used by the CHILI and KAGNN baselines: 425 graphs
per crystal system (2,975 graphs), drawn with numpy.default_rng(42) from the metadata index
of the full CHILI-100K release, and split 80/10/10 with sklearn.train_test_split
(stratified by space-group number, random_state=42).

The subset is stored in one HDF5 file (see `write_benchmark`) so that training and
evaluation never need the 14.5 GB source archive.
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile
from collections import Counter

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .config import (CRYSTAL_SYSTEMS, MODEL, OUT_DIM, SCATTERING, SPLIT_SEED,
                     SUBSET_PER_CLASS, ModelConfig)
from .model import rbf_encode

SYS_TO_IDX = {s: i for i, s in enumerate(CRYSTAL_SYSTEMS)}
BENCHMARK_FILE = "chili100k_benchmark.h5"


# ============================================================= index of the full release
def _size_sort_key(k):
    return float(re.findall(r"[\d.]+", k)[0])


def _index_file(f, name):
    g = f["GlobalLabels"]
    cs = g["CrystalSystem"][()]
    cs = cs.decode() if isinstance(cs, (bytes, bytearray)) else str(cs)
    cs = SYS_TO_IDX.get(cs, -1)
    sg = int(g["SpaceGroupNumber"][()])
    keys = sorted(f["DiscreteParticleGraphs"].keys(), key=_size_sort_key)
    return [(name, k, cs, sg, rank) for rank, k in enumerate(keys)]


class Source:
    """Read-only access to a CHILI-100K release: the zip archive or its extracted folder.
    Member names are paths relative to the archive root, e.g. 'cod_output_080124/1000007.h5'."""

    def __init__(self, path):
        self.path = path
        self.zip = zipfile.ZipFile(path) if os.path.isfile(path) else None

    def members(self):
        if self.zip is not None:
            names = [n for n in self.zip.namelist() if n.endswith(".h5")]
        else:
            names = [os.path.relpath(os.path.join(d, f), self.path)
                     for d, _, fs in os.walk(self.path) for f in fs if f.endswith(".h5")]
        return sorted(names)

    def open(self, name):
        if self.zip is not None:
            return h5py.File(io.BytesIO(self.zip.read(name)), "r")
        return h5py.File(os.path.join(self.path, name), "r")


def scan_index(source: Source, log_every=2000):
    """One entry per (material file, particle size): (file, size_key, crystal_system,
    space_group, size_rank), files in sorted path order, sizes in ascending order."""
    index = []
    names = source.members()
    for j, name in enumerate(names):
        try:
            with source.open(name) as f:
                index += _index_file(f, name)
        except Exception:
            pass
        if log_every and (j + 1) % log_every == 0:
            print(f"  scanned {j + 1}/{len(names)}", flush=True)
    return index


def select_benchmark(index, per_class=SUBSET_PER_CLASS, seed=SPLIT_SEED):
    """Stratified subset: `per_class` graphs per crystal system, returned in index order."""
    rng = np.random.default_rng(seed)
    by_cs = {}
    for i, rec in enumerate(index):
        if rec[2] >= 0:
            by_cs.setdefault(rec[2], []).append(i)
    picked = []
    for c in sorted(by_cs):
        pool = np.array(by_cs[c])
        picked += rng.choice(pool, size=min(per_class, len(pool)), replace=False).tolist()
    return [index[i] for i in sorted(picked)]


def official_split(space_groups, seed=SPLIT_SEED):
    """80/10/10 split over individual graphs, stratified by space-group number."""
    idx = np.arange(len(space_groups))
    strat = np.asarray(space_groups)
    _, c = np.unique(strat, return_counts=True)
    if (c < 2).any():
        strat = None
    tr, te = train_test_split(idx, test_size=0.10, random_state=seed, stratify=strat)
    strat_tr = strat[tr] if strat is not None else None
    tr, va = train_test_split(tr, test_size=0.10 / 0.90, random_state=seed, stratify=strat_tr)
    return tr.tolist(), va.tolist(), te.tolist()


def read_graph(f, size_key):
    """The arrays the model uses for one particle size of one material file."""
    d = f["DiscreteParticleGraphs"][size_key]
    rec = dict(nf=d["NodeFeatures"][:].astype(np.float32),
               ei=d["EdgeDirections"][:].astype(np.int64),
               pos=d["AbsoluteCoordinates"][:].astype(np.float32))
    sc = f["ScatteringData"][size_key]
    rec["scatter"] = {t: sc[fld][1].astype(np.float32) for t, fld in SCATTERING.items()}
    rec["axes"] = {t: sc[fld][0].astype(np.float64) for t, fld in SCATTERING.items()}
    return rec


# ============================================================= benchmark file (HDF5)
def write_benchmark(path, entries, records, axes, full_crystal_system_counts, split):
    """entries[i] = (file, size_key, cs, sg, rank); records[i] = dict(nf, ei, pos, scatter)."""
    G = len(entries)
    n_nodes = np.array([records[i]["nf"].shape[0] for i in range(G)], dtype=np.int64)
    n_edges = np.array([records[i]["ei"].shape[1] for i in range(G)], dtype=np.int64)
    node_ptr = np.concatenate([[0], np.cumsum(n_nodes)])
    edge_ptr = np.concatenate([[0], np.cumsum(n_edges)])
    comp = dict(compression="gzip", compression_opts=4, shuffle=True)
    with h5py.File(path, "w") as f:
        f.attrs["description"] = ("CHILI-100K benchmark subset: 425 graphs per crystal system "
                                  "(2,975 graphs), numpy.default_rng(42); official 80/10/10 split "
                                  "(sklearn.train_test_split, stratified by space group, random_state=42)")
        f.attrs["n_graphs"] = G
        f.attrs["full_dataset_crystal_system_counts"] = json.dumps(full_crystal_system_counts)
        ix = f.create_group("index")
        ix.create_dataset("source_file", data=[e[0] for e in entries], dtype=h5py.string_dtype())
        ix.create_dataset("size_key", data=[e[1] for e in entries], dtype=h5py.string_dtype())
        ix.create_dataset("crystal_system", data=np.array([e[2] for e in entries], dtype=np.int8))
        ix.create_dataset("space_group", data=np.array([e[3] for e in entries], dtype=np.int16))
        ix.create_dataset("size_rank", data=np.array([e[4] for e in entries], dtype=np.int8))
        sp = f.create_group("split")
        for k in ("train", "val", "test"):
            sp.create_dataset(k, data=np.array(split[k], dtype=np.int32))
        g = f.create_group("graph")
        g.create_dataset("node_ptr", data=node_ptr)
        g.create_dataset("edge_ptr", data=edge_ptr)
        g.create_dataset("node_features", data=np.concatenate([records[i]["nf"] for i in range(G)]), **comp)
        g.create_dataset("positions", data=np.concatenate([records[i]["pos"] for i in range(G)]), **comp)
        g.create_dataset("edge_index", data=np.concatenate([records[i]["ei"] for i in range(G)],
                                                           axis=1).astype(np.int32), **comp)
        s = f.create_group("scattering")
        for t in SCATTERING:
            s.create_dataset(t, data=np.stack([records[i]["scatter"][t] for i in range(G)]), **comp)
            s.create_dataset(f"{t}_axis", data=np.asarray(axes[t], dtype=np.float64))


class Benchmark:
    """In-memory view of the benchmark file.

    entries[i]  (file, size_key, crystal_system, space_group, size_rank)
    records[i]  dict(nf (N,4) float32, ei (2,E) int64, pos (N,3) float32, cs, sg, rank,
                     scatter={saxs, xrd, xpdf}) -- intensities only; axes in `axes`
    split       dict(train, val, test) -> positions into entries/records
    """

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            ix = f["index"]
            files = list(ix["source_file"].asstr()[:])
            keys = list(ix["size_key"].asstr()[:])
            cs, sg, rk = ix["crystal_system"][:], ix["space_group"][:], ix["size_rank"][:]
            self.entries = [(files[i], keys[i], int(cs[i]), int(sg[i]), int(rk[i]))
                            for i in range(len(files))]
            self.split = {k: f["split"][k][:].tolist() for k in ("train", "val", "test")}
            self.full_crystal_system_counts = json.loads(f.attrs["full_dataset_crystal_system_counts"])
            g = f["graph"]
            node_ptr, edge_ptr = g["node_ptr"][:], g["edge_ptr"][:]
            nf, pos, ei = g["node_features"][:], g["positions"][:], g["edge_index"][:]
            sc = {t: f["scattering"][t][:] for t in SCATTERING}
            self.axes = {t: f["scattering"][f"{t}_axis"][:] for t in SCATTERING}
        self.records = []
        for i, (_, _, c, s, r) in enumerate(self.entries):
            a, b = node_ptr[i], node_ptr[i + 1]
            e0, e1 = edge_ptr[i], edge_ptr[i + 1]
            self.records.append(dict(nf=nf[a:b], pos=pos[a:b],
                                     ei=ei[:, e0:e1].astype(np.int64),
                                     cs=c, sg=s, rank=r,
                                     scatter={t: sc[t][i] for t in SCATTERING}))

    def __len__(self):
        return len(self.records)

    def check_split(self):
        """The stored split must be the one official_split() produces from the space groups."""
        tr, va, te = official_split([e[3] for e in self.entries])
        return tr == self.split["train"] and va == self.split["val"] and te == self.split["test"]


def load_benchmark(path=None):
    path = path or os.path.join("data", BENCHMARK_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run `python scripts/download.py --data` "
                                f"or build it with scripts/build_benchmark.py")
    return Benchmark(path)


# ============================================================= label spaces
def label_space(task, bench: Benchmark, sg_vocab="all"):
    """Output width and (for space group) the label map.

    sg_vocab="all"   : 230 classes, label = space-group number - 1
    sg_vocab="train" : one class per space group present in the training split
    """
    if task != "space_group":
        return OUT_DIM[task], None
    if sg_vocab == "all":
        return 230, {s: s - 1 for s in range(1, 231)}
    classes = sorted(set(int(bench.records[i]["sg"]) for i in bench.split["train"]))
    return len(classes), {s: i for i, s in enumerate(classes)}


def sg_vocab_of_checkpoint(out_dim):
    return "all" if out_dim == 230 else "train"


# ============================================================= PyG dataset
class CHILIDataset(Dataset):
    """Graph `i` of the benchmark as a PyG Data object.

    node features   non-atom tasks: z = atomic number, x = [radius, weight, e-affinity] (raw;
                    scaled by BatchNorm in the model); atom task: z = 0, x = centred coords / 10
    edge features   RBF(bond length, 32 functions over 0-6 Å), unit vector, raw length
    graph features  [log1p(n_atoms), n_species, size_rank] (raw; BatchNorm in the model)
    targets         crystal system; space group; element per atom; scattering curves
                    min-max normalised per sample to [0, 1]

    `node_drop` (training only): each graph with more than 24 atoms loses each atom with
    probability node_drop, together with its bonds, provided at least 12 atoms remain.
    """

    def __init__(self, records, task, sg_to_idx=None, node_drop=0.0, cfg: ModelConfig = MODEL):
        self.records, self.task, self.sg_to_idx = records, task, sg_to_idx
        self.node_drop, self.cfg = node_drop, cfg

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        nf = torch.from_numpy(r["nf"])
        pos = torch.from_numpy(r["pos"])
        ei = torch.from_numpy(r["ei"]).long()

        if self.node_drop > 0 and nf.shape[0] > 24:
            keep = torch.rand(nf.shape[0]) > self.node_drop
            if int(keep.sum()) >= 12:
                remap = torch.full((nf.shape[0],), -1, dtype=torch.long)
                remap[keep] = torch.arange(int(keep.sum()))
                emask = keep[ei[0]] & keep[ei[1]]
                ei = remap[ei[:, emask]]
                nf = nf[keep]
                pos = pos[keep]

        posc = pos - pos.mean(0, keepdim=True)
        if ei.numel() == 0:
            ei = torch.zeros(2, 1, dtype=torch.long)
        vec = posc[ei[1]] - posc[ei[0]]
        rd = vec.norm(dim=-1).clamp(min=1e-6)

        if self.task == "atom":
            z = torch.zeros(nf.shape[0], dtype=torch.long)
            cont = posc / self.cfg.pos_scale
        else:
            z = nf[:, 0].long().clamp(0, 118)
            cont = nf[:, 1:4]

        ga = torch.tensor([np.log1p(nf.shape[0]), float(len(torch.unique(nf[:, 0]))),
                           float(r["rank"])], dtype=torch.float32).unsqueeze(0)

        d = Data(x=cont, z=z, edge_index=ei,
                 edge_attr=rbf_encode(rd, self.cfg.num_rbf_pos, 0.0, self.cfg.rbf_cutoff),
                 edge_unit_vec=vec / rd.unsqueeze(-1),
                 edge_raw_dist=rd, graph_attr=ga, pos=posc)
        if self.task == "crystal_system":
            d.y = torch.tensor([max(r["cs"], 0)], dtype=torch.long)
        elif self.task == "space_group":
            d.y = torch.tensor([self.sg_to_idx.get(int(r["sg"]), -1)], dtype=torch.long)
        elif self.task == "atom":
            d.y = nf[:, 0].long().clamp(1, 118) - 1
        else:
            it = torch.from_numpy(r["scatter"][self.task])
            d.y = ((it - it.min()) / (it.max() - it.min() + 1e-9)).unsqueeze(0)
        return d


# ============================================================= split file (tracked in git)
def write_split_json(path, entries, split, full_counts):
    doc = {"description": "CHILI-100K benchmark subset (425 graphs per crystal system) and "
                          "official 80/10/10 split. graphs[i] = [source_file, size_key, "
                          "crystal_system (0-6), space_group (1-230), size_rank (0-4)].",
           "crystal_systems": list(CRYSTAL_SYSTEMS),
           "full_dataset_crystal_system_counts": full_counts,
           "n_graphs": len(entries),
           "graphs": [list(e) for e in entries],
           "train": split["train"], "val": split["val"], "test": split["test"]}
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))


def read_split_json(path):
    with open(path) as f:
        doc = json.load(f)
    entries = [tuple(e) for e in doc["graphs"]]
    return entries, {k: doc[k] for k in ("train", "val", "test")}, doc["full_dataset_crystal_system_counts"]


def crystal_system_counts(index):
    c = Counter(r[2] for r in index if r[2] >= 0)
    return {CRYSTAL_SYSTEMS[k]: int(c.get(k, 0)) for k in range(len(CRYSTAL_SYSTEMS))}
