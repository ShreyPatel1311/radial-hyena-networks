"""Read-out operator extraction: per-branch Jacobians and spline nonlinearity.

For a KANLinear block with input x, the mean Jacobian splits by branch:

    J_base[k,j]   = W_base[k,j] * E_n[ SiLU'(x_nj) ]
    J_spline[k,j] = sum_c w[k,j,c] * E_n[ B'_c(x_nj) ]

The spline branch is linear in the B-spline basis, so the expectation factors through the
basis derivative and only the (in_features, n_coeff) mean basis-derivative is needed.
"""
from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def collect_kan_inputs(model, loader, device, max_rows=20000):
    """Capture the input to each KAN block, plus the graph id of every row."""
    blocks = list(model.kan.blocks)
    store = {i: [] for i in range(len(blocks))}
    gids, hooks, cache = [], [], {}

    def mk(i):
        def hook(mod, inp, out):
            cache[i] = inp[0].detach().reshape(-1, mod.in_features)
        return hook

    for i, b in enumerate(blocks):
        hooks.append(b.register_forward_hook(mk(i)))

    model.eval()
    gid0 = 0
    for bt in loader:
        bt = bt.to(device)
        model(bt)
        n = cache[0].shape[0]
        nb = int(bt.batch.max().item()) + 1 if getattr(bt, "batch", None) is not None else 1
        gids.append(np.arange(gid0, gid0 + nb) if n == nb else gid0 + bt.batch.cpu().numpy())
        gid0 += nb
        for i in range(len(blocks)):
            store[i].append(cache[i].float().cpu())
        cache.clear()
    for h in hooks:
        h.remove()

    X = {i: torch.cat(v) for i, v in store.items()}
    g = np.concatenate(gids)
    if X[0].shape[0] > max_rows:
        sel = np.linspace(0, X[0].shape[0] - 1, max_rows).astype(int)
        X = {i: v[sel] for i, v in X.items()}
        g = g[sel]
    return X, g


@torch.no_grad()
def block_operator(block, x, device, eps=1e-3, n_grid=41, want_curves=False,
                   chunk_out=512):
    """Mean per-branch Jacobian and per-edge nonlinearity for one KANLinear block.

    NL[k,j] = 1 - R^2 of the best affine fit to phi_kj over the empirical input range of
    unit j, with grid points placed at quantiles of x_j.
    """
    x = x.to(device)
    w = block.spline_weight.data
    Wb = block.base_weight.data

    s = torch.sigmoid(x)
    dsilu = (s * (1 + x * (1 - s))).mean(0)
    J_base = Wb * dsilu.unsqueeze(0)

    dB = ((block.b_splines(x + eps) - block.b_splines(x - eps)) / (2 * eps)).mean(0)
    J_spline = torch.einsum("oic,ic->oi", w, dB)

    qs = torch.linspace(0.02, 0.98, n_grid, device=device)
    Xg = torch.quantile(x, qs, dim=0)
    Bg = block.b_splines(Xg)
    NL = torch.zeros(block.out_features, block.in_features, device=device)
    curves = torch.zeros(block.out_features, block.in_features, n_grid) if want_curves else None
    t = Xg - Xg.mean(0, keepdim=True)
    tvar = (t ** 2).sum(0).clamp(min=1e-12)
    for o0 in range(0, block.out_features, chunk_out):
        ws = w[o0:o0 + chunk_out]
        phi = torch.einsum("gic,oic->ogi", Bg, ws)
        if want_curves:
            curves[o0:o0 + chunk_out] = phi.permute(0, 2, 1).cpu()
        phic = phi - phi.mean(1, keepdim=True)
        slope = (phic * t.unsqueeze(0)).sum(1) / tvar.unsqueeze(0)
        resid = phic - slope.unsqueeze(1) * t.unsqueeze(0)
        NL[o0:o0 + chunk_out] = (resid ** 2).sum(1) / (phic ** 2).sum(1).clamp(min=1e-12)

    return dict(J_base=J_base.cpu().numpy(), J_spline=J_spline.cpu().numpy(),
                NL=NL.cpu().numpy(), grid_x=Xg.cpu().numpy(),
                curves=(curves.numpy() if want_curves else None))


def kan_edge_count(model):
    return int(sum(b.out_features * b.in_features for b in model.kan.blocks))


def apply_spline_mask(model, orig_spline, keep_flat):
    """Zero the spline branch of the de-selected edges, leaving the base branch intact."""
    off = 0
    for b, w0 in zip(model.kan.blocks, orig_spline):
        n = b.out_features * b.in_features
        m = torch.from_numpy(keep_flat[off:off + n].astype(np.float32))
        b.spline_weight.data = w0 * m.view(b.out_features, b.in_features, 1).to(w0.device)
        off += n
    assert off == keep_flat.size, "mask length does not match the KAN edge count"


def restore_splines(model, orig_spline, orig_base=None):
    for i, b in enumerate(model.kan.blocks):
        b.spline_weight.data = orig_spline[i].clone()
        if orig_base is not None:
            b.base_weight.data = orig_base[i].clone()


@torch.no_grad()
def spline_importances(model, loader, device, max_batches=None):
    """Per-edge spline importance under two criteria.

    weight_l2  : ||spline_weight[i,j,:]||_2
    activation : mean over evaluated samples of |phi_ij(x_j)|
    """
    blocks = list(model.kan.blocks)
    acc = [torch.zeros(b.out_features, b.in_features, device=device) for b in blocks]
    cnt = 0
    hooks, cache = [], {}

    def mk(i):
        def hook(mod, inp, out):
            cache[i] = inp[0].detach().reshape(-1, mod.in_features)
        return hook

    for i, b in enumerate(blocks):
        hooks.append(b.register_forward_hook(mk(i)))

    model.eval()
    for nb, bt in enumerate(loader):
        if max_batches is not None and nb >= max_batches:
            break
        model(bt.to(device))
        for i, b in enumerate(blocks):
            x = cache[i]
            step = int(min(2048, max(8, 2e7 // (b.out_features * b.in_features))))
            for c0 in range(0, x.shape[0], step):
                bs = b.b_splines(x[c0:c0 + step])
                acc[i] += torch.einsum("nik,oik->noi", bs, b.spline_weight).abs().sum(0)
            if i == 0:
                cnt += x.shape[0]
        cache.clear()
    for h in hooks:
        h.remove()

    w_l2 = torch.cat([b.spline_weight.norm(dim=-1).flatten() for b in blocks]).cpu()
    act = torch.cat([(a / max(cnt, 1)).flatten() for a in acc]).cpu()
    return {"weight_l2": w_l2.numpy(), "activation": act.numpy()}


@torch.no_grad()
def coarsen_spline_grid(model, orig_spline, g_new, n_pts=400):
    """Refit every spline onto a coarser B-spline grid by least squares."""
    from .kan import KANLinear
    for b, w0 in zip(model.kan.blocks, orig_spline):
        dev = w0.device
        lo = b.grid[:, b.spline_order].clone()
        hi = b.grid[:, -(b.spline_order + 1)].clone()
        t = torch.linspace(0, 1, n_pts, device=dev).unsqueeze(-1)
        x = lo.unsqueeze(0) + t * (hi - lo).unsqueeze(0)
        y = torch.einsum("nik,oik->nio", b.b_splines(x), w0)
        coarse = KANLinear(b.in_features, b.out_features, grid_size=g_new,
                           spline_order=b.spline_order).to(dev)
        b.grid = coarse.grid
        b.spline_weight = torch.nn.Parameter(coarse.spline_weight.data)
        b.grid_size = g_new
        b.spline_weight.data = b.curve2coeff(x, y)


@torch.no_grad()
def restore_spline_grid(model, orig_grid, orig_spline, orig_grid_size):
    for i, b in enumerate(model.kan.blocks):
        b.grid = orig_grid[i].clone()
        b.grid_size = orig_grid_size[i]
        b.spline_weight = torch.nn.Parameter(orig_spline[i].clone())
