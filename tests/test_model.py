import os
import sys

import pytest
import torch
from torch_geometric.data import Batch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena.config import OUT_DIM, level_of               # noqa: E402
from radial_hyena.data import CHILIDataset                      # noqa: E402
from radial_hyena.model import RadialHyena, count_parameters    # noqa: E402

# parameter counts of the released checkpoints
PARAMS = {("crystal_system", 7): 220_878, ("space_group", 230): 349_326,
          ("space_group", 151): 303_822, ("atom", 118): 284_808,
          ("saxs", 300): 389_646, ("xrd", 580): 550_926, ("xpdf", 6000): 3_672_846}


@pytest.mark.parametrize("task,out_dim", list(PARAMS))
def test_parameter_counts(task, out_dim):
    model = RadialHyena(out_dim, level=level_of(task))
    assert count_parameters(model) == PARAMS[(task, out_dim)]


def _record(n, rng):
    pos = torch.randn(n, 3, generator=rng) * 4
    d = torch.cdist(pos, pos)
    src, dst = torch.nonzero((d < 3.5) & (d > 0), as_tuple=True)
    nf = torch.stack([torch.randint(1, 90, (n,), generator=rng).float(),
                      torch.rand(n, generator=rng), torch.rand(n, generator=rng) * 200,
                      torch.rand(n, generator=rng)], 1)
    return dict(nf=nf.numpy(), pos=pos.numpy(), ei=torch.stack([src, dst]).numpy(),
                cs=3, sg=225, rank=1,
                scatter={t: torch.rand(OUT_DIM[t], generator=rng).numpy() for t in ("saxs", "xrd", "xpdf")})


def _batch(task, sizes=(40, 57, 33)):
    rng = torch.Generator().manual_seed(0)
    recs = [_record(n, rng) for n in sizes]
    sg_to_idx = {s: s - 1 for s in range(1, 231)} if task == "space_group" else None
    ds = CHILIDataset(recs, task, sg_to_idx)
    return Batch.from_data_list([ds[i] for i in range(len(ds))]), sum(sizes)


@pytest.mark.parametrize("task", ["crystal_system", "space_group", "atom", "saxs"])
def test_forward_shapes(task):
    bt, n_atoms = _batch(task)
    model = RadialHyena(OUT_DIM[task], level=level_of(task)).eval()
    with torch.no_grad():
        out = model(bt)
    rows = n_atoms if task == "atom" else 3
    assert out.shape == (rows, OUT_DIM[task])
    assert torch.isfinite(out).all()


def test_readout_input_layout():
    bt, _ = _batch("crystal_system")
    model = RadialHyena(7).eval()
    with torch.no_grad():
        h_v, h_e, g, batch, B, layers = model.trunk(bt, keep_layers=True)
        x = model.readout_input(h_v, h_e, g, batch, B, bt.edge_index)
        out = model.kan(x)
    assert x.shape == (3, 192) and len(layers) == 3
    assert torch.allclose(out, model(bt), atol=1e-6)


def test_cpu_forward_is_deterministic():
    bt, _ = _batch("saxs")
    model = RadialHyena(300).eval()
    with torch.no_grad():
        assert torch.equal(model(bt), model(bt))
