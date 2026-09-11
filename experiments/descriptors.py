#!/usr/bin/env python3
"""Physical descriptors of the test particles, computed from their 3-D coordinates.

  R_g, R_max          radius of gyration; distance of the outermost atom from the centroid
  sphericity          R_g / R_max
  surface_to_volume   3 / R_max (sphere)
  number_density      N / (4/3 pi R_max^3)
  bond length         mean and standard deviation over the graph's bonds
  coordination        mean and standard deviation of the bond count per atom
  first_shell_r       first peak of g(r) inside 1-6 Å
  mean_Z              mean atomic number
  Q4, Q6, Q8          Steinhardt bond-orientational order parameters
  (+ n_atoms, log_n_atoms, n_species, size_rank, crystal_system, space_group as bookkeeping)

Output: results/analysis/descriptors_test.npz (names, values, pair_hist, r_centers, test_index)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DESCRIPTORS                                  # noqa: E402
from radial_hyena.data import load_benchmark                     # noqa: E402

PAIR_CAP = 1200           # atoms sampled for the pair-distance histogram
PDF_MAX, PDF_BINS = 30.0, 300
STEIN_CAP = 600           # central atoms sampled for the Steinhardt sums
STEIN_L = (4, 6, 8)


def pair_distance_hist(pos, rng, cap=PAIR_CAP, rmax=PDF_MAX, nbins=PDF_BINS):
    p = pos
    if p.shape[0] > cap:
        p = p[rng.choice(p.shape[0], cap, replace=False)]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    d = d[np.triu_indices(p.shape[0], k=1)]
    h, edges = np.histogram(d, bins=nbins, range=(0.0, rmax))
    return h.astype(np.float64) / max(h.sum(), 1), 0.5 * (edges[1:] + edges[:-1])


def steinhardt(pos, ei, cap=STEIN_CAP, ls=STEIN_L):
    """Q_l averaged over sampled central atoms, using the graph's own bonds."""
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
            Y = sph_harm_y(l, m[:, None], theta[None, :], phi[None, :])
            qlm = Y.mean(axis=1)
            out[f"Q{l}"].append(np.sqrt(4 * np.pi / (2 * l + 1) * np.sum(np.abs(qlm) ** 2)))
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in out.items()}


def descriptors_for(rec, rng):
    pos = rec["pos"].astype(np.float64)
    ei, nf = rec["ei"], rec["nf"]
    N = pos.shape[0]
    dr = np.linalg.norm(pos - pos.mean(0), axis=-1)
    Rg, Rmax = float(np.sqrt((dr ** 2).mean())), float(dr.max())
    src, dst = ei[0], ei[1]
    self_loop = src == dst
    bl = (np.linalg.norm(pos[dst[~self_loop]] - pos[src[~self_loop]], axis=-1)
          if (~self_loop).any() else np.array([np.nan]))
    deg = np.bincount(src[~self_loop], minlength=N)
    P, r_centers = pair_distance_hist(pos, rng)
    # first coordination shell: peak of g(r) = P(r) / r^2 within 1-6 Å
    win = (r_centers >= 1.0) & (r_centers <= 6.0)
    g = np.zeros_like(P)
    nz = r_centers > 1e-6
    g[nz] = P[nz] / (r_centers[nz] ** 2)
    first = r_centers[win][np.argmax(g[win])] if g[win].sum() > 0 else np.nan
    d = dict(n_atoms=float(N), log_n_atoms=float(np.log1p(N)), R_g=Rg, R_max=Rmax,
             sphericity=float(Rg / max(Rmax, 1e-9)),
             surface_to_volume=float(3.0 / max(Rmax, 1e-9)),
             number_density=float(N / max((4 / 3) * np.pi * Rmax ** 3, 1e-9)),
             mean_bond_length=float(np.nanmean(bl)), std_bond_length=float(np.nanstd(bl)),
             mean_coordination=float(deg.mean()), std_coordination=float(deg.std()),
             first_shell_r=float(first), n_species=float(len(np.unique(nf[:, 0]))),
             mean_Z=float(np.mean(nf[:, 0])), size_rank=float(rec["rank"]),
             crystal_system=float(rec["cs"]), space_group=float(rec["sg"]))
    d.update(steinhardt(pos, ei))
    return d, P, r_centers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/chili100k_benchmark.h5")
    ap.add_argument("--out", default=DESCRIPTORS)
    a = ap.parse_args()
    bench = load_benchmark(a.data)
    te = bench.split["test"]
    rng = np.random.default_rng(0)
    rows, Ps, t0 = [], [], time.time()
    for n, i in enumerate(te):
        d, P, rc = descriptors_for(bench.records[i], rng)
        rows.append(d)
        Ps.append(P)
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(te)} ({time.time() - t0:.0f}s)", flush=True)
    names = list(rows[0])
    M = np.array([[r[k] for k in names] for r in rows], dtype=np.float64)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, names=np.array(names), values=M, pair_hist=np.array(Ps),
             r_centers=rc, test_index=np.array(te))
    print(f"{M.shape[0]} test particles x {M.shape[1]} descriptors -> {a.out}")


if __name__ == "__main__":
    main()
