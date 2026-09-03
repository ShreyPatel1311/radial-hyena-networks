#!/usr/bin/env python3
"""Compute physical descriptors from the 3D coordinates of each particle.

    python experiments/physics_descriptors.py --split test

Each descriptor relates to an equation linking structure to a measured target:
  R_g            Guinier:  I(q) ~ I(0) exp(-q^2 R_g^2 / 3)
  S/V, R_max     Porod:    I(q) ~ 2 pi (S/V) q^-4
  P(r)           Debye:    I(q) = sum_ij f_i f_j sinc(q r_ij)
  Q4, Q6, Q8     Steinhardt bond-orientational order parameters
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena import data as D
from radial_hyena.physics import descriptors_for
from radial_hyena.runner import load_task


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--data-zip", default=None)
    ap.add_argument("--split", default="test", choices=["test", "all"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/physics_descriptors_{a.split}.npz"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    sub, (tr, va, te), order, sg, records, _ = load_task("crystal_system", a.cache_dir,
                                                          a.data_zip, a.split)
    smap = {i: "train" for i in tr}
    smap.update({i: "val" for i in va}); smap.update({i: "test" for i in te})

    rng = np.random.default_rng(0)
    rows, hists = [], []
    t0 = time.time()
    for n, i in enumerate(order):
        d, P, rc = descriptors_for(records[i], rng)
        rows.append(d); hists.append(P)
        if (n + 1) % 250 == 0:
            print(f"  {n + 1}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)

    names = list(rows[0].keys())
    M = np.array([[r[k] for k in names] for r in rows], dtype=np.float64)
    np.savez(out, names=np.array(names), values=M, pair_hist=np.array(hists),
             r_centers=rc, index=np.array(order),
             split=np.array([smap[i] for i in order]))
    print(f"\n{M.shape[0]} particles x {M.shape[1]} descriptors -> {out}")
    for j, k in enumerate(names):
        v = M[:, j]
        print(f"  {k:20s} median={np.nanmedian(v):10.4f}  "
              f"[{np.nanmin(v):9.3f}, {np.nanmax(v):10.3f}]  nan={int(np.isnan(v).sum())}")


if __name__ == "__main__":
    main()
