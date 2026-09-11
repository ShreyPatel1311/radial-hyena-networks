"""B-spline Kolmogorov-Arnold layers (efficient-KAN formulation)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """y_o = sum_i [ W_base[o,i] * SiLU(x_i) + sum_c W_spline[o,i,c] * B_c(x_i) ]

    B_c are cubic B-splines on a uniform grid over `grid_range`, extended by
    `spline_order` knots on each side (grid_size + spline_order coefficients per edge).
    """

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=nn.SiLU, grid_range=(-1., 1.)):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
                .expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features,
                                                      grid_size + spline_order))
        self.scale_base, self.scale_spline, self.scale_noise = scale_base, scale_spline, scale_noise
        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                     * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(self.scale_spline * self.curve2coeff(
                self.grid.T[self.spline_order:-self.spline_order], noise))

    def b_splines(self, x):
        """(n, in) -> (n, in, grid_size + spline_order) basis values (Cox-de Boor)."""
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]) \
                  + ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        sol = torch.linalg.lstsq(A, B).solution
        return sol.permute(2, 0, 1).contiguous()

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, self.in_features)
        base = F.linear(self.base_activation(x), self.base_weight)
        spline = F.linear(self.b_splines(x).view(x.size(0), -1),
                          self.spline_weight.view(self.out_features, -1))
        return (base + spline).reshape(*shp[:-1], self.out_features)


class KAN(nn.Module):
    """Stack of KANLinear layers; dropout between layers."""

    def __init__(self, layers, grid_size=5, spline_order=3, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([KANLinear(a, b, grid_size=grid_size, spline_order=spline_order)
                                     for a, b in zip(layers[:-1], layers[1:])])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i < len(self.blocks) - 1:
                x = self.dropout(x)
        return x
