"""Download the benchmark data, checkpoints and training records.

The files live in a Hugging Face repository with the layout

    data/chili100k_benchmark.h5
    checkpoints/<task>_seed<k>.pt
    records/<training records>
    MANIFEST.md5

and every file is verified against the MD5 checksums in MANIFEST.md5.

By default files are fetched through the anonymous mirror of that repository (plain HTTPS,
no extra dependencies); the mirror id is ANON_ID below, overridable with the environment
variable RADIAL_HYENA_ANON_ID. Passing `repo_id` fetches from a Hugging Face repository
directly instead, which requires `huggingface_hub`. Files downloaded by hand can be checked
with verify_local() and installed with `scripts/download.py --from-dir`.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.parse
import urllib.request

ANON_ID = os.environ.get("RADIAL_HYENA_ANON_ID", "jp5aih03acut")
ANON_API = "https://anonymous-hf.com/api/a"
REPO_TYPE = "model"
MANIFEST = "MANIFEST.md5"
# The mirror sits behind a filter that rejects urllib's default "Python-urllib/x.y" agent.
USER_AGENT = "radial-hyena"


def anon_url(relpath, anon_id=ANON_ID):
    return f"{ANON_API}/{anon_id}/resolve/{urllib.parse.quote(relpath)}"


def _open(url, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
                                  timeout=timeout)


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


def _hf_download(repo_id, relpath, revision):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("downloading from a Hugging Face repository needs huggingface_hub: "
                          "pip install huggingface_hub") from e
    return hf_hub_download(repo_id, relpath, repo_type=REPO_TYPE, revision=revision)


def read_manifest(anon_id=ANON_ID, repo_id=None, revision=None):
    """Download MANIFEST.md5 and parse it."""
    if repo_id:
        with open(_hf_download(repo_id, MANIFEST, revision)) as f:
            return parse_manifest(f.read())
    with _open(anon_url(MANIFEST, anon_id), timeout=60) as r:
        return parse_manifest(r.read().decode())


def fetch(relpath, dest_dir, manifest, anon_id=ANON_ID, repo_id=None, revision=None,
          overwrite=False):
    """Download one file into dest_dir and verify its MD5. Returns the local path."""
    if relpath not in manifest:
        raise FileNotFoundError(f"{relpath} is not listed in {MANIFEST}")
    expected = manifest[relpath]
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, os.path.basename(relpath))
    if os.path.exists(path) and not overwrite and md5(path) == expected:
        return path
    tmp = path + ".part"
    if repo_id:
        shutil.copyfile(_hf_download(repo_id, relpath, revision), tmp)
    else:
        with _open(anon_url(relpath, anon_id)) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
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
