#!/usr/bin/env python3
"""Fetch the benchmark data, released checkpoints and training records from Hugging Face.

    python scripts/download.py --all            # everything (~190 MB)
    python scripts/download.py --data           # data/chili100k_benchmark.h5
    python scripts/download.py --checkpoints    # checkpoints/*.pt (21 files)
    python scripts/download.py --records        # results/training/ (training histories)

The repository is set in radial_hyena/hub.py (or export RADIAL_HYENA_HF_REPO=...).
Every file is checked against the MD5 checksums in the repository's MANIFEST.md5.

Files downloaded by hand, e.g. through a browser, can be verified and put in place with

    python scripts/download.py --from-dir /path/to/downloaded/repository
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena import hub                                                 # noqa: E402
from radial_hyena.config import TASKS                                        # noqa: E402

SEEDS = (0, 1, 2)
DATA = "data/chili100k_benchmark.h5"
CHECKPOINTS = ([f"checkpoints/{t}_seed{s}.pt" for t in TASKS for s in SEEDS]
               + [f"checkpoints/space_group_trainvocab_seed{s}.pt" for s in SEEDS])
DEST = {"data": "data", "checkpoints": "checkpoints", "records": os.path.join("results", "training")}


def _selected(a, manifest):
    todo = []
    if a.all or a.data:
        todo.append(DATA)
    if a.all or a.checkpoints:
        todo += CHECKPOINTS
    if a.all or a.records:
        todo += sorted(p for p in manifest if p.startswith("records/"))
    return todo


def _install_from_dir(root):
    status = hub.verify_local(root)
    bad = {p: s for p, s in status.items() if s != "ok"}
    for p, s in sorted(bad.items()):
        print(f"  {s:8s} {p}")
    if bad:
        raise SystemExit(f"{len(bad)} of {len(status)} files failed verification; nothing installed")
    for rel in sorted(status):
        dest = DEST[rel.split("/", 1)[0]]
        os.makedirs(dest, exist_ok=True)
        target = os.path.join(dest, os.path.basename(rel))
        shutil.copyfile(os.path.join(root, rel), target)
        print(f"  {target}")
    print(f"verified and installed {len(status)} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--checkpoints", action="store_true")
    ap.add_argument("--records", action="store_true")
    ap.add_argument("--repo", default=hub.REPO_ID, help="Hugging Face repository id")
    ap.add_argument("--revision", default=None, help="branch, tag or commit of the repository")
    ap.add_argument("--from-dir", default=None,
                    help="verify a hand-downloaded copy of the repository and install it")
    a = ap.parse_args()

    if a.from_dir:
        _install_from_dir(a.from_dir)
        return
    if not (a.all or a.data or a.checkpoints or a.records):
        ap.error("choose --all, --data, --checkpoints, --records or --from-dir")

    manifest = hub.read_manifest(a.repo, a.revision)
    for rel in _selected(a, manifest):
        path = hub.fetch(rel, DEST[rel.split("/", 1)[0]], manifest, a.repo, a.revision)
        print(f"  {path}")


if __name__ == "__main__":
    main()
