import os
import sys
from collections import Counter

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from radial_hyena import data as D                     # noqa: E402
from radial_hyena.config import SUBSET_PER_CLASS       # noqa: E402

SPLIT_JSON = os.path.join(ROOT, "splits", "chili100k_benchmark_split.json")
BENCH = os.path.join(ROOT, "data", D.BENCHMARK_FILE)


@pytest.fixture(scope="module")
def split_file():
    return D.read_split_json(SPLIT_JSON)


def test_subset_composition(split_file):
    entries, split, full = split_file
    assert len(entries) == 2975
    assert Counter(e[2] for e in entries) == {c: SUBSET_PER_CLASS for c in range(7)}
    assert sum(full.values()) == 104_408
    assert len(set(entries)) == len(entries)


def test_split_sizes_and_disjointness(split_file):
    entries, split, _ = split_file
    tr, va, te = map(set, (split["train"], split["val"], split["test"]))
    assert (len(tr), len(va), len(te)) == (2379, 298, 298)
    assert not (tr & va or tr & te or va & te)
    assert tr | va | te == set(range(len(entries)))


def test_split_is_reproducible_from_space_groups(split_file):
    entries, split, _ = split_file
    tr, va, te = D.official_split([e[3] for e in entries])
    assert (tr, va, te) == (split["train"], split["val"], split["test"])


def test_select_benchmark_on_synthetic_index():
    rng = np.random.default_rng(1)
    index = [(f"f{i // 5}.h5", f"{i % 5}", int(rng.integers(0, 7)), 1 + int(rng.integers(0, 230)), i % 5)
             for i in range(7000)]
    sub = D.select_benchmark(index, per_class=50)
    assert Counter(e[2] for e in sub) == {c: 50 for c in range(7)}
    assert sub == sorted(sub, key=index.index)                 # kept in index order


def _record(n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g) * 4
    d = torch.cdist(pos, pos)
    src, dst = torch.nonzero((d < 3.5) & (d > 0), as_tuple=True)
    nf = torch.stack([torch.randint(1, 90, (n,), generator=g).float(), torch.rand(n, generator=g),
                      torch.rand(n, generator=g), torch.rand(n, generator=g)], 1)
    return dict(nf=nf.numpy(), pos=pos.numpy(), ei=torch.stack([src, dst]).numpy(), cs=2, sg=62, rank=3,
                scatter={"saxs": np.linspace(5, 9, 300, dtype=np.float32),
                         "xrd": np.ones(580, np.float32), "xpdf": np.ones(6000, np.float32)})


def test_dataset_targets_and_features():
    r = _record()
    d = D.CHILIDataset([r], "saxs")[0]
    assert d.y.shape == (1, 300) and float(d.y.min()) == 0.0 and abs(float(d.y.max()) - 1.0) < 1e-6
    assert d.graph_attr.shape == (1, 3) and float(d.graph_attr[0, 2]) == 3.0
    assert d.edge_attr.shape[1] == 32 and torch.allclose(d.pos.mean(0), torch.zeros(3), atol=1e-5)
    a = D.CHILIDataset([r], "atom")[0]
    assert (a.z == 0).all() and a.y.shape == (60,)
    assert torch.equal(a.y, torch.from_numpy(r["nf"][:, 0]).long().clamp(1, 118) - 1)


def test_node_drop_only_when_requested():
    r = _record(n=200)
    torch.manual_seed(0)
    full = D.CHILIDataset([r], "crystal_system")[0]
    dropped = D.CHILIDataset([r], "crystal_system", node_drop=0.2)[0]
    assert full.num_nodes == 200 and 12 <= dropped.num_nodes < 200
    assert int(dropped.edge_index.max()) < dropped.num_nodes


@pytest.mark.skipif(not os.path.exists(BENCH), reason="data/chili100k_benchmark.h5 not downloaded")
def test_benchmark_file_matches_split_file(split_file):
    entries, split, _ = split_file
    b = D.load_benchmark(BENCH)
    assert b.entries == entries and b.split == split and b.check_split()
