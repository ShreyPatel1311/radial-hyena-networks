"""Architecture invariants required for the released checkpoints to load."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from radial_hyena import RadialHyenaNet

EXPECTED_PARAMS = {
    ("crystal_system", 7, "graph"): 220878,
    ("space_group", 151, "graph"): 303822,
    ("atom", 118, "node"): 284808,
    ("saxs", 300, "graph"): 389646,
    ("xrd", 580, "graph"): 550926,
    ("xpdf", 6000, "graph"): 3672846,
}


@pytest.mark.parametrize("spec,n_expected", list(EXPECTED_PARAMS.items()))
def test_parameter_counts(spec, n_expected):
    task, out_dim, level = spec
    model = RadialHyenaNet(out_dim, level=level)
    assert sum(p.numel() for p in model.parameters()) == n_expected


def test_kan_block_shapes():
    model = RadialHyenaNet(7, "graph")
    b0, b1 = model.kan.blocks
    assert (b0.in_features, b0.out_features) == (192, 64)
    assert (b1.in_features, b1.out_features) == (64, 7)
    assert b0.spline_weight.shape == (64, 192, 8)


def test_spline_and_base_branches_are_separable():
    model = RadialHyenaNet(7, "graph")
    b = model.kan.blocks[0]
    base_before = b.base_weight.data.clone()
    b.spline_weight.data.zero_()
    assert torch.equal(b.base_weight.data, base_before)
