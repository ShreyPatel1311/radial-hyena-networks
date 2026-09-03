#!/usr/bin/env python3
"""Extract the physical axes of the scattering targets.

    python experiments/physical_axes.py --data-zip /path/to/CHILI-100K.zip

  SAXS  q = 0 .. 2.99   A^-1, 300 points
  XRD   q = 1 .. 29.95  A^-1, 580 points
  xPDF  r = 0 .. 59.99  A,   6000 points
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena import data as D


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-zip", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results/physical_axes.npz")
    ap.add_argument("--check", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    index = D.build_index(a.data_zip, cache=os.path.join(a.cache_dir, "chili_index.pkl"))
    sub = D.benchmark_subset(index)
    axes = {}
    for field in ("SAXS", "XRD", "xPDF"):
        ref = D.scattering_axis(a.data_zip, sub, field)
        same = all(np.array_equal(ref, D.scattering_axis(a.data_zip, sub[i:], field))
                   for i in range(1, a.check))
        axes[field] = ref
        print(f"{field:5s} n={len(ref):5d} range=[{ref[0]:g}, {ref[-1]:g}] "
              f"step={ref[1] - ref[0]:g} | identical across {a.check} structures: {same}")
    np.savez(a.out, **axes)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
