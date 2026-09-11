#!/usr/bin/env python3
"""Fetch the benchmark data, released checkpoints and training records from Dataverse.

    python scripts/download.py --all            # everything (~200 MB)
    python scripts/download.py --data           # data/chili100k_benchmark.h5
    python scripts/download.py --checkpoints    # checkpoints/*.pt (21 files)
    python scripts/download.py --records        # results/training/*.json (training histories)

The dataset DOI is set in radial_hyena/dataverse.py (or export RADIAL_HYENA_DOI=...).
Every file is checked against the MD5 checksum Dataverse reports for it.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena import dataverse                                           # noqa: E402
from radial_hyena.config import TASKS                                        # noqa: E402

SEEDS = (0, 1, 2)
CHECKPOINTS = ([f"{t}_seed{s}.pt" for t in TASKS for s in SEEDS]
               + [f"space_group_trainvocab_seed{s}.pt" for s in SEEDS])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--checkpoints", action="store_true")
    ap.add_argument("--records", action="store_true")
    ap.add_argument("--doi", default=dataverse.DATASET_DOI)
    a = ap.parse_args()
    if not (a.all or a.data or a.checkpoints or a.records):
        ap.error("choose --all, --data, --checkpoints or --records")

    files = dataverse.list_files(a.doi)
    todo = []
    if a.all or a.data:
        todo.append(("chili100k_benchmark.h5", "data"))
    if a.all or a.checkpoints:
        todo += [(n, "checkpoints") for n in CHECKPOINTS]
    if a.all or a.records:
        todo += [(n, os.path.join("results", "training")) for n in sorted(files)
                 if n.endswith(("_results.json", "_history.json", ".log"))]
    for name, dest in todo:
        path = dataverse.fetch(name, dest, files, a.doi)
        print(f"  {path}")


if __name__ == "__main__":
    main()
