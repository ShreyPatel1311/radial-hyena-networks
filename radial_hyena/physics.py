"""
Physical descriptors for the CHILI-100K test particles, computed from the REAL 3D
coordinates -- these are the quantities the governing physics actually depends on.

Each descriptor is tied to a specific equation relating structure to a target:

  R_g            Guinier:  I(q) ~ I(0) exp(-q^2 R_g^2 / 3)  for q R_g < 1.3
                 -> the sole structural parameter controlling low-q SAXS.
  S/V, R_max     Porod:    I(q) ~ 2 pi (S/V) q^-4 at high q -> surface scattering.
  P(r)           Debye:    I(q) = sum_ij f_i f_j sinc(q r_ij)
                 -> the pair-distance distribution IS the input to the scattering
                    equation; xPDF G(r) is essentially P(r) itself.
  d-spacings     Bragg:    q_hkl = 2 pi / d_hkl -> XRD peak positions.
  Q4, Q6, Q8     Steinhardt bond-orientational order parameters,
                 Q_l = sqrt(4 pi/(2l+1) * sum_m |<Y_lm>|^2)
                 -> the standard order parameters that discriminate crystal symmetry
                    (fcc/bcc/hcp/sc have distinct, well-known Q4/Q6 values).

Used by the descriptor-coupling and spline-physics experiments.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
from scipy.special import sph_harm_y

PAIR_CAP = 1200          # atoms sampled for the O(N^2) pair-distance histogram
PDF_MAX, PDF_BINS = 30.0, 300
STEIN_CAP = 600          # central atoms sampled for Steinhardt sums
STEIN_L = (4, 6, 8)


def pair_distance_hist(pos, rng, cap=PAIR_CAP, rmax=PDF_MAX, nbins=PDF_BINS):
    """Normalised pair-distance distribution P(r) -- the Debye equation's structural input."""
    p = pos
    if p.shape[0] > cap:
        p = p[rng.choice(p.shape[0], cap, replace=False)]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    d = d[np.triu_indices(p.shape[0], k=1)]
    h, edges = np.histogram(d, bins=nbins, range=(0.0, rmax))
    return h.astype(np.float64) / max(h.sum(), 1), 0.5 * (edges[1:] + edges[:-1])


def steinhardt(pos, ei, cap=STEIN_CAP, ls=STEIN_L):
    """Q_l for l in ls, averaged over sampled central atoms, using the graph's own bonds."""
    N = pos.shape[0]
    src, dst = ei[0], ei[1]
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    starts = np.searchsorted(src, np.arange(N), side="left")
    ends = np.searchsorted(src, np.arange(N), side="right")
    deg = ends - starts
    cand = np.where(deg >= 3)[0]
    if cand.size == 0:
        return {f"Q{l}": np.nan for l in ls}
    if cand.size > cap:
        cand = cand[np.linspace(0, cand.size - 1, cap).astype(int)]
    out = {f"Q{l}": [] for l in ls}
    for i in cand:
        nb = dst[starts[i]:ends[i]]
        v = pos[nb] - pos[i]
        r = np.linalg.norm(v, axis=-1)
        keep = r > 1e-6
        v, r = v[keep], r[keep]
        if v.shape[0] < 3:
            continue
        theta = np.arccos(np.clip(v[:, 2] / r, -1, 1))
        phi = np.arctan2(v[:, 1], v[:, 0])
        for l in ls:
            m = np.arange(-l, l + 1)
            # sph_harm_y(n, m, theta, phi) -> broadcast over (m, neighbours)
            Y = sph_harm_y(l, m[:, None], theta[None, :], phi[None, :])
            qlm = Y.mean(axis=1)
            out[f"Q{l}"].append(np.sqrt(4 * np.pi / (2 * l + 1) * np.sum(np.abs(qlm) ** 2)))
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in out.items()}


def descriptors_for(rec, rng):
    pos = rec["pos"].astype(np.float64)
    ei = rec["ei"]
    nf = rec["nf"]
    N = pos.shape[0]
    cen = pos.mean(0)
    dr = np.linalg.norm(pos - cen, axis=-1)
    Rg = float(np.sqrt((dr ** 2).mean()))
    Rmax = float(dr.max())

    src, dst = ei[0], ei[1]
    self_loop = src == dst
    bl = np.linalg.norm(pos[dst[~self_loop]] - pos[src[~self_loop]], axis=-1) if (~self_loop).any() \
        else np.array([np.nan])
    deg = np.bincount(src[~self_loop], minlength=N)

    P, r_centers = pair_distance_hist(pos, rng)
    # first coordination shell = the P(r) peak inside the bonding range. The search MUST be
    # windowed: in a 10,000-atom particle the global argmax of P(r) sits out at the bulk
    # pair separation (~14 A), not at the first shell.
    # ... and it must be found in g(r), not P(r): the pair COUNT grows as r^2 by geometry,
    # so P(r) alone still peaks at the top of any window. g(r) = P(r) / (4 pi r^2 rho)
    # removes that trivial growth and leaves the actual coordination shells.
    win = (r_centers >= 1.0) & (r_centers <= 6.0)
    g = np.zeros_like(P)
    nz = r_centers > 1e-6
    g[nz] = P[nz] / (r_centers[nz] ** 2)
    first = r_centers[win][np.argmax(g[win])] if g[win].sum() > 0 else np.nan

    d = dict(
        n_atoms=float(N),
        log_n_atoms=float(np.log1p(N)),
        R_g=Rg,
        R_max=Rmax,
        sphericity=float(Rg / max(Rmax, 1e-9)),          # 0.775 for a uniform sphere
        surface_to_volume=float(3.0 / max(Rmax, 1e-9)),   # Porod S/V for a sphere
        number_density=float(N / max((4 / 3) * np.pi * Rmax ** 3, 1e-9)),
        mean_bond_length=float(np.nanmean(bl)),
        std_bond_length=float(np.nanstd(bl)),
        mean_coordination=float(deg.mean()),
        std_coordination=float(deg.std()),
        first_shell_r=float(first),
        n_species=float(len(np.unique(nf[:, 0]))),
        mean_Z=float(np.mean(nf[:, 0])),
        size_rank=float(rec["rank"]),
        crystal_system=float(rec["cs"]),
        space_group=float(rec["sg"]),
    )
    d.update(steinhardt(pos, ei, ))
    return d, P, r_centers
