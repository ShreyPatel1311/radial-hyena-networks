"""Radial Hyena: node (radial) + edge (bond) Hyena streams and a graph MLP, read out by a KAN.

    node stream   one sequence per particle; the implicit long-convolution filter at each
                  position is generated from an RBF expansion of the atom's normalised
                  distance to the particle centroid, r / r_max (0 = centre, 1 = outermost atom)
    edge stream   per-atom sequences of up to `max_k` bonds; filter generated from the RBF
                  expansion of the bond length, content = bond embedding + distance + angle
    graph stream  MLP over [log1p(n_atoms), n_species, size_rank]

Node and edge features are pooled with learned softmax gates (graph-level tasks) and
concatenated with the graph vector: [node_pool | edge_pool | graph] -> KAN -> output.
For the node-level atom task the KAN reads [h_v | mean incident edge | graph] per atom.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MODEL, ModelConfig
from .kan import KAN


# --------------------------------------------------------------------- basis encodings
def rbf_encode(x, num_rbf, lo, hi):
    c = torch.linspace(lo, hi, num_rbf, device=x.device)
    s = (hi - lo) / num_rbf
    return torch.exp(-((x.unsqueeze(-1) - c) ** 2) / (2 * s ** 2))


def angle_encode(cos, num_basis):
    theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
    n = torch.arange(num_basis, device=cos.device, dtype=torch.float32)
    return torch.cos(n * theta.unsqueeze(-1))


# --------------------------------------------------------------------- scatter helpers
def scatter_sum(src, index, dim_size, dim=0):
    shape = list(src.shape)
    shape[dim] = dim_size
    out = src.new_zeros(shape)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(dim, idx, src)


def scatter_mean(src, index, dim_size, dim=0):
    s = scatter_sum(src, index, dim_size, dim)
    c = scatter_sum(torch.ones_like(src[..., :1]), index, dim_size, dim).clamp(min=1)
    return s / c


def scatter_softmax(src, index, dim_size, dim=0):
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    mx = src.new_full([dim_size, *src.shape[1:]], float("-inf"))
    mx.scatter_reduce_(dim, idx, src, reduce="amax", include_self=True)
    ex = (src - mx.gather(dim, idx)).exp()
    den = scatter_sum(ex, index, dim_size, dim).gather(dim, idx).clamp(min=1e-16)
    return ex / den


# --------------------------------------------------------------------- Hyena primitives
class Sin(nn.Module):
    def __init__(self, dim, w=10.0):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim))

    def forward(self, x):
        return torch.sin(self.freq * x)


def fftconv(u, k, D):
    """Causal linear convolution via FFT (length rounded up to a power of two >= 2L-1)."""
    L = u.shape[-1]
    n = 1 << max(1, 2 * L - 1).bit_length()
    k_f = torch.fft.rfft(k, n=n) / n
    u_f = torch.fft.rfft(u, n=n)
    y = torch.fft.irfft(u_f * k_f, n=n, norm="forward")[..., :L]
    return (y + u * D.unsqueeze(-1)).to(u.dtype)


class LongConv(nn.Module):
    """Hyena long convolution whose filter is generated implicitly from a conditioning basis."""

    def __init__(self, cond_dim, seq_dim, filter_hidden=64):
        super().__init__()
        self.implicit_filter = nn.Sequential(
            nn.Linear(cond_dim, filter_hidden), Sin(filter_hidden),
            nn.Linear(filter_hidden, filter_hidden), Sin(filter_hidden),
            nn.Linear(filter_hidden, seq_dim, bias=False))
        self.D = nn.Parameter(torch.zeros(seq_dim))

    def forward(self, seq, cond, mask):
        m = mask.float().unsqueeze(-1)
        h = self.implicit_filter(cond) * m
        seq = seq * m
        with torch.autocast(device_type=seq.device.type, enabled=False):
            out = fftconv(seq.permute(0, 2, 1).float(), h.permute(0, 2, 1).float(), self.D.float())
        return out.permute(0, 2, 1).to(seq.dtype) * m


# --------------------------------------------------------------------- streams
class RadialNodeStream(nn.Module):
    """One sequence per particle; filter conditioned on the normalised radius r / r_max.

    Sequence positions come from a stable argsort of the float32 key
    `graph_index * 1e9 + r`, the key the released weights were trained with.
    """

    def __init__(self, dim, num_rbf=32, filter_hidden=64, inject_pos=True):
        super().__init__()
        self.num_rbf = num_rbf
        self.inject_pos = inject_pos
        if inject_pos:
            self.pos_embed = nn.Linear(3, dim)      # additive embedding of centred coordinates
        self.conv = LongConv(num_rbf, dim, filter_hidden)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, pos, batch, B):
        if self.inject_pos:
            h = h + self.pos_embed(pos)
        N, d = h.shape
        dev = h.device
        cen = scatter_mean(pos, batch, B)
        r = (pos - cen[batch]).norm(dim=-1)
        rmax = torch.zeros(B, device=dev).scatter_reduce(0, batch, r, reduce="amax",
                                                         include_self=False).clamp(min=1e-6)
        rn = (r / rmax[batch]).clamp(0, 1)
        counts = torch.bincount(batch, minlength=B)
        Nmax = int(counts.max().item())
        offs = F.pad(counts.cumsum(0), (1, 0))
        order = torch.argsort(batch.float() * 1e9 + r, stable=True)
        gs = batch[order]
        rank = torch.arange(N, device=dev) - offs[gs]
        flat = gs * Nmax + rank
        seq = h.new_zeros(B * Nmax, d)
        seq[flat] = h[order]
        cond = h.new_zeros(B * Nmax, self.num_rbf)
        cond[flat] = rbf_encode(rn[order], self.num_rbf, 0.0, 1.0)
        mask = torch.zeros(B * Nmax, dtype=torch.bool, device=dev)
        mask[flat] = True
        out = self.conv(seq.view(B, Nmax, d), cond.view(B, Nmax, self.num_rbf), mask.view(B, Nmax))
        h_out = h.new_zeros(N, d)
        h_out[order] = out.reshape(B * Nmax, d)[flat]
        return self.norm(h_out)


class EdgeHyenaStream(nn.Module):
    """Per-atom bond sequences (grouped by source atom, up to `max_k` bonds per atom).

    Bond positions come from a stable argsort of the float32 key
    `source_atom * 1e7 + bond_length`. Each element is [bond embedding | RBF(d) | angle
    basis of the bond against the first bond in the atom's sequence].
    """

    def __init__(self, edge_dim, num_rbf_pos=32, num_rbf_ang=16, max_k=12, filter_hidden=64,
                 cutoff=6.0):
        super().__init__()
        self.max_k, self.nrp, self.nra, self.cutoff = max_k, num_rbf_pos, num_rbf_ang, cutoff
        d_seq = edge_dim + num_rbf_pos + num_rbf_ang
        self.conv = LongConv(num_rbf_pos, d_seq, filter_hidden)
        self.proj = nn.Linear(d_seq, edge_dim)
        self.norm = nn.LayerNorm(edge_dim)

    def forward(self, h_e, edge_index, unit_vec, edge_raw, N):
        dev = h_e.device
        E = h_e.shape[0]
        keep = edge_index[0] != edge_index[1]
        gidx = torch.where(keep)[0]
        grp = edge_index[0][gidx]
        ordv = edge_raw[gidx]
        s = torch.argsort(grp.float() * 1e7 + ordv, stable=True)
        gs, feat_s, uv_s = grp[s], h_e[gidx][s], unit_vec[gidx][s]
        d_s = edge_raw[gidx][s]
        counts = torch.bincount(gs, minlength=N)
        offs = F.pad(counts.cumsum(0), (1, 0))
        row = torch.arange(gs.shape[0], device=dev) - offs[gs]
        valid = row < self.max_k
        orig = gidx[s[valid]]
        flat = gs[valid] * self.max_k + row[valid]
        cond_v = rbf_encode(d_s[valid], self.nrp, 0.0, self.cutoff)
        first = uv_s[offs[gs[valid]]]
        ang = angle_encode((uv_s[valid] * first).sum(-1).clamp(-1, 1), self.nra)
        elem = torch.cat([feat_s[valid], cond_v, ang], dim=-1)
        d_seq = elem.shape[-1]
        seq = h_e.new_zeros(N * self.max_k, d_seq)
        seq[flat] = elem
        cond = h_e.new_zeros(N * self.max_k, self.nrp)
        cond[flat] = cond_v
        mask = torch.zeros(N * self.max_k, dtype=torch.bool, device=dev)
        mask[flat] = True
        out = self.conv(seq.view(N, self.max_k, d_seq), cond.view(N, self.max_k, self.nrp),
                        mask.view(N, self.max_k)).reshape(N * self.max_k, d_seq)
        delta = h_e.new_zeros(E, d_seq)
        delta[orig] = out[flat]
        return self.norm(h_e + self.proj(delta))


class GraphStream(nn.Module):
    def __init__(self, graph_in, dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(graph_in, dim), nn.SiLU(),
                                 nn.Linear(dim, dim), nn.SiLU())

    def forward(self, graph_attr):
        return self.mlp(graph_attr)


# --------------------------------------------------------------------- full model
class RadialHyena(nn.Module):
    def __init__(self, out_dim, level="graph", cfg: ModelConfig = MODEL):
        super().__init__()
        self.level, self.cfg = level, cfg
        # input scaling: BatchNorm on the raw node scalars and graph attributes (running
        # statistics accumulate on training batches); atom-task coordinates use pos_scale
        self.node_bn = nn.Identity() if level == "node" else nn.BatchNorm1d(3)
        self.graph_bn = nn.BatchNorm1d(cfg.graph_in)
        self.z_embed = nn.Embedding(119, 32)
        self.node_embed = nn.Sequential(nn.Linear(32 + 3, cfg.node_dim * 2), nn.SiLU(),
                                        nn.Linear(cfg.node_dim * 2, cfg.node_dim))
        self.edge_embed = nn.Sequential(nn.Linear(cfg.num_rbf_pos, cfg.edge_dim * 2), nn.SiLU(),
                                        nn.Linear(cfg.edge_dim * 2, cfg.edge_dim))
        self.node_layers = nn.ModuleList([
            RadialNodeStream(cfg.node_dim, cfg.num_rbf_pos, cfg.filter_hidden, cfg.inject_pos)
            for _ in range(cfg.num_layers)])
        self.node_norms = nn.ModuleList([nn.LayerNorm(cfg.node_dim) for _ in range(cfg.num_layers)])
        self.edge_layers = nn.ModuleList([
            EdgeHyenaStream(cfg.edge_dim, cfg.num_rbf_pos, cfg.num_rbf_ang, cfg.max_k,
                            cfg.filter_hidden, cfg.rbf_cutoff)
            for _ in range(cfg.num_layers)])
        self.graph_stream = GraphStream(cfg.graph_in, cfg.graph_dim)
        self.node_gate = nn.Sequential(nn.Linear(cfg.node_dim, cfg.node_dim), nn.Tanh(),
                                       nn.Linear(cfg.node_dim, 1))
        self.edge_gate = nn.Sequential(nn.Linear(cfg.edge_dim, cfg.edge_dim), nn.Tanh(),
                                       nn.Linear(cfg.edge_dim, 1))
        kin = cfg.node_dim + cfg.edge_dim + cfg.graph_dim
        layers = [kin, out_dim] if not cfg.kan_hidden else [kin, cfg.kan_hidden, out_dim]
        self.kan = KAN(layers, grid_size=cfg.kan_grid, spline_order=cfg.spline_order,
                       dropout=cfg.dropout)

    @staticmethod
    def _batch_of(data, n):
        b = getattr(data, "batch", None)
        return b if b is not None else torch.zeros(n, dtype=torch.long, device=data.x.device)

    def trunk(self, data, keep_layers=False):
        """Embeddings -> Hyena layers -> graph stream. Returns (h_v, h_e, g, batch, B, layers)."""
        h_v = self.node_embed(torch.cat([self.z_embed(data.z), self.node_bn(data.x)], dim=-1))
        h_e = self.edge_embed(data.edge_attr)
        ei, uv = data.edge_index, data.edge_unit_vec
        batch = self._batch_of(data, h_v.shape[0])
        B, N = int(batch.max()) + 1, h_v.shape[0]
        layers = [h_v] if keep_layers else None
        for nl, nn_, el_ in zip(self.node_layers, self.node_norms, self.edge_layers):
            h_v = h_v + nl(nn_(h_v), data.pos, batch, B)
            h_e = el_(h_e, ei, uv, data.edge_raw_dist, N)
            if keep_layers:
                layers.append(h_v)
        g = self.graph_stream(self.graph_bn(data.graph_attr))
        return h_v, h_e, g, batch, B, layers

    def readout_input(self, h_v, h_e, g, batch, B, edge_index):
        """The concatenated vector the KAN reads: graph level (B, 192) or node level (N, 192)."""
        if self.level == "node":
            e2n = scatter_mean(h_e, edge_index[1], h_v.shape[0])
            return torch.cat([h_v, e2n, g[batch]], dim=-1)
        nv = scatter_softmax(self.node_gate(h_v), batch, B)
        node_pool = scatter_sum(h_v * nv, batch, B)
        eb = batch[edge_index[0]]
        ev = scatter_softmax(self.edge_gate(h_e), eb, B)
        edge_pool = scatter_sum(h_e * ev, eb, B)
        return torch.cat([node_pool, edge_pool, g], dim=-1)

    def forward(self, data):
        h_v, h_e, g, batch, B, _ = self.trunk(data)
        return self.kan(self.readout_input(h_v, h_e, g, batch, B, data.edge_index))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
