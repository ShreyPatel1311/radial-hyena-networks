"""The analytic mean Jacobian must agree with autograd."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from radial_hyena.kan import KANLinear
from radial_hyena.operator import block_operator, coarsen_spline_grid, restore_spline_grid


def test_analytic_jacobian_matches_autograd():
    torch.manual_seed(0)
    block = KANLinear(12, 5)
    x = torch.randn(1, 12) * 0.3
    J_auto = torch.autograd.functional.jacobian(lambda z: block(z).squeeze(0), x).squeeze()
    r = block_operator(block, x, "cpu")
    J_analytic = torch.from_numpy(r["J_base"] + r["J_spline"])
    rel = (J_auto - J_analytic).abs().max() / J_auto.abs().max()
    assert rel < 1e-3, f"relative error {rel:.2e}"


def test_grid_coarsening_at_original_size_is_near_identity():
    class Wrapper(torch.nn.Module):
        def __init__(self, blk):
            super().__init__()
            self.kan = torch.nn.Module()
            self.kan.blocks = torch.nn.ModuleList([blk])

    torch.manual_seed(0)
    block = KANLinear(8, 4, grid_size=5)
    model = Wrapper(block)
    orig = [block.spline_weight.data.clone()]
    coarsen_spline_grid(model, orig, 5)
    x = torch.rand(16, 8) * 1.2 - 0.6
    refit = block(x)
    restore_spline_grid(model, [block.grid.clone()], orig, [5])
    assert torch.allclose(refit, block(x), atol=5e-2)
