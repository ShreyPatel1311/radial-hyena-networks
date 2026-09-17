"""Download the benchmark data, checkpoints and training records from Hugging Face.

The files live in a Hugging Face repository with the layout

    data/chili100k_benchmark.h5
    checkpoints/<task>_seed<k>.pt
    records/<training records>
    MANIFEST.md5

and every file is verified against the MD5 checksums in MANIFEST.md5.

The repository id defaults to REPO_ID below; override it with the environment variable
RADIAL_HYENA_HF_REPO. Files downloaded by hand (for instance through a browser) can be
checked with verify_local() and installed with `scripts/download.py --from-dir`.
"""
from __future__ import annotations

import hashlib
import os
import shutil

REPO_ID = os.environ.get("RADIAL_HYENA_HF_REPO", "Godseye1311/radial-hyena-networks")
REPO_TYPE = "model"
MANIFEST = "MANIFEST.md5"


def parse_manifest(text):
    """{relative path: md5} from the text of an `md5sum`-style manifest."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, path = line.split(None, 1)
        out[path.strip().lstrip("*").removeprefix("./")] = digest.lower()
    return out


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_manifest(repo_id=REPO_ID, revision=None):
    """Download MANIFEST.md5 from the repository and parse it."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, MANIFEST, repo_type=REPO_TYPE, revision=revision)
    with open(path) as f:
        return parse_manifest(f.read())


def fetch(relpath, dest_dir, manifest, repo_id=REPO_ID, revision=None, overwrite=False):
    """Download one file into dest_dir and verify its MD5. Returns the local path.

    Private repositories need a token, taken from HF_TOKEN or `huggingface-cli login`.
    """
    from huggingface_hub import hf_hub_download
    if relpath not in manifest:
        raise FileNotFoundError(f"{relpath} is not listed in {MANIFEST} of {repo_id}")
    expected = manifest[relpath]
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, os.path.basename(relpath))
    if os.path.exists(path) and not overwrite and md5(path) == expected:
        return path
    cached = hf_hub_download(repo_id, relpath, repo_type=REPO_TYPE, revision=revision)
    tmp = path + ".part"
    shutil.copyfile(cached, tmp)
    if md5(tmp) != expected:
        os.remove(tmp)
        raise IOError(f"checksum mismatch for {relpath}")
    os.replace(tmp, path)
    return path


def verify_local(root):
    """Check a hand-downloaded copy of the repository against its own MANIFEST.md5.

    Returns {relative path: "ok" | "missing" | "mismatch"}.
    """
    with open(os.path.join(root, MANIFEST)) as f:
        manifest = parse_manifest(f.read())
    status = {}
    for rel, expected in manifest.items():
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            status[rel] = "missing"
        else:
            status[rel] = "ok" if md5(p) == expected else "mismatch"
    return status
