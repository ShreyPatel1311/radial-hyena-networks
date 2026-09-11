#!/usr/bin/env python3
"""Build (or verify) the benchmark file from an original CHILI-100K release.

    python scripts/build_benchmark.py --source /path/to/CHILI-100K.zip
    python scripts/build_benchmark.py --source /path/to/CHILI-100K/ --verify data/chili100k_benchmark.h5

Build: scans every material file of the release (sorted by path), selects 425 graphs per
crystal system with numpy.default_rng(42), applies the official 80/10/10 split and writes
data/chili100k_benchmark.h5. The selection and split are checked against the tracked
splits/chili100k_benchmark_split.json.

Verify: compares every graph of an existing benchmark file with the release, array by array.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena import data as D                                           # noqa: E402
from radial_hyena.config import SCATTERING                                   # noqa: E402

SPLIT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "splits", "chili100k_benchmark_split.json")


def read_records(src, entries, log_every=250):
    records, axes = [], None
    for n, (name, key, cs, sg, rank) in enumerate(entries):
        with src.open(name) as f:
            rec = D.read_graph(f, key)
        axes = axes or rec["axes"]
        rec.update(cs=cs, sg=sg, rank=rank)
        records.append(rec)
        if log_every and (n + 1) % log_every == 0:
            print(f"  read {n + 1}/{len(entries)}", flush=True)
    return records, axes


def verify(src, path):
    bench = D.load_benchmark(path)
    ok = bench.check_split()
    print(f"split == official split: {ok}")
    t0 = time.time()
    for i, (name, key, cs, sg, rank) in enumerate(bench.entries):
        with src.open(name) as f:
            rec = D.read_graph(f, key)
            labels = D._index_file(f, name)
        r = bench.records[i]
        same = (all(np.array_equal(rec[k], r[k]) for k in ("nf", "ei", "pos"))
                and all(np.array_equal(rec["scatter"][t], r["scatter"][t]) for t in SCATTERING)
                and all(np.allclose(rec["axes"][t], bench.axes[t]) for t in SCATTERING)
                and (name, key, cs, sg, rank) in labels)
        if not same:
            print(f"  MISMATCH graph {i}: {name} {key}")
            ok = False
        if (i + 1) % 250 == 0:
            print(f"  verified {i + 1}/{len(bench)} ({time.time() - t0:.0f}s)", flush=True)
    print("benchmark file identical to the release:" if ok else "differences found:", ok)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="CHILI-100K.zip or its extracted folder")
    ap.add_argument("--out", default=os.path.join("data", D.BENCHMARK_FILE))
    ap.add_argument("--verify", metavar="BENCHMARK_H5", help="compare an existing benchmark file instead")
    a = ap.parse_args()
    src = D.Source(a.source)
    if a.verify:
        sys.exit(0 if verify(src, a.verify) else 1)

    t0 = time.time()
    index = D.scan_index(src)
    print(f"index: {len(index)} graphs ({time.time() - t0:.0f}s)")
    entries = D.select_benchmark(index)
    tr, va, te = D.official_split([e[3] for e in entries])
    split = {"train": tr, "val": va, "test": te}
    ref_entries, ref_split, _ = D.read_split_json(SPLIT_JSON)
    assert [tuple(e) for e in entries] == [tuple(e) for e in ref_entries], "subset differs from splits/*.json"
    assert split == ref_split, "split differs from splits/*.json"
    print(f"subset: {len(entries)} graphs, split {len(tr)}/{len(va)}/{len(te)} (matches {os.path.basename(SPLIT_JSON)})")
    records, axes = read_records(src, entries)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    D.write_benchmark(a.out, entries, records, axes, D.crystal_system_counts(index), split)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.0f} MB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
