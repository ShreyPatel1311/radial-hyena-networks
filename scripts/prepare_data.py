#!/usr/bin/env python3
"""Build the metadata index and cache the graph payloads.

    python scripts/prepare_data.py --data-zip /path/to/CHILI-100K.zip

Run once before training or analysis. Every later script reuses the cache.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena import data as D


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-zip", required=True, help="path to CHILI-100K.zip")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--no-records", action="store_true", help="build the index only")
    a = ap.parse_args()
    os.makedirs(a.cache_dir, exist_ok=True)

    t0 = time.time()
    index = D.build_index(a.data_zip, cache=os.path.join(a.cache_dir, "chili_index.pkl"))
    print(f"index: {len(index):,} graphs ({time.time() - t0:.0f}s)")

    sub = D.benchmark_subset(index)
    tr, va, te = D.make_split(sub)
    voc = D.space_group_vocab(sub, tr)
    unmap = sum(1 for i in te if int(sub[i][3]) not in voc)
    print(f"benchmark subset: {len(sub):,} graphs -> train {len(tr)} / val {len(va)} / test {len(te)}")
    print(f"space-group classes (train-only): {len(voc)} | test graphs outside vocab: {unmap}")
    expected = (2975, 2379, 298, 298, 151, 3)
    got = (len(sub), len(tr), len(va), len(te), len(voc), unmap)
    print(f"matches the benchmark protocol: {got == expected}"
          + ("" if got == expected else f"  (expected {expected}, got {got})"))

    if not a.no_records:
        t0 = time.time()
        D.load_records(sub, sorted(tr + va + te), a.data_zip,
                       cache=os.path.join(a.cache_dir, "records.pt"))
        print(f"cached graph payloads ({time.time() - t0:.0f}s)")
    print(f"\ncache ready in {a.cache_dir}/")


if __name__ == "__main__":
    main()
