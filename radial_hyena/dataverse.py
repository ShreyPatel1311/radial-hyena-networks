"""Download files from the project's Harvard Dataverse dataset (native API, no extra deps).

The dataset DOI defaults to DATASET_DOI below; override per call with the environment
variable RADIAL_HYENA_DOI.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request

SERVER = os.environ.get("DATAVERSE_SERVER", "https://dataverse.harvard.edu")
DATASET_DOI = os.environ.get("RADIAL_HYENA_DOI", "doi:10.7910/DVN/QC34B9")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "radial-hyena"})
    token = os.environ.get("DATAVERSE_API_TOKEN")        # only needed for draft/private versions
    if token:
        req.add_header("X-Dataverse-key", token)
    return urllib.request.urlopen(req, timeout=120)


def list_files(doi=DATASET_DOI, server=SERVER, version=":latest"):
    """{filename: {"id", "md5", "size", "dir"}} for every file in the dataset version."""
    q = urllib.parse.urlencode({"persistentId": doi})
    with _get(f"{server}/api/datasets/:persistentId/versions/{version}/files?{q}") as r:
        data = json.load(r)["data"]
    out = {}
    for item in data:
        df = item["dataFile"]
        md5 = df.get("md5") or (df.get("checksum", {}).get("value")
                                if df.get("checksum", {}).get("type") == "MD5" else None)
        name = df.get("originalFileName") or df["filename"]
        out[name] = {"id": df["id"], "md5": md5, "size": df.get("originalFileSize", df.get("filesize")),
                     "dir": item.get("directoryLabel", ""),
                     "original": "originalFileFormat" in df}
    return out


def _md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def upload(path, directory_label=None, description=None, doi=DATASET_DOI, server=SERVER,
          token=None, timeout=1800):
    """Add one local file to the dataset's draft version, verified by MD5 after upload.

    Requires a Dataverse API token with write access to this dataset, via `token` or the
    DATAVERSE_API_TOKEN environment variable. Depends on `requests` (not a base dependency
    of this package, since only this function needs multipart upload).
    """
    import requests
    token = token or os.environ.get("DATAVERSE_API_TOKEN")
    if not token:
        raise RuntimeError("no Dataverse API token: pass token= or set DATAVERSE_API_TOKEN")
    meta = {}
    if directory_label:
        meta["directoryLabel"] = directory_label
    if description:
        meta["description"] = description
    url = f"{server}/api/datasets/:persistentId/add"
    with open(path, "rb") as f:
        r = requests.post(url, params={"persistentId": doi},
                          headers={"X-Dataverse-key": token},
                          data={"jsonData": json.dumps(meta)},
                          files={"file": (os.path.basename(path), f)}, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    if body.get("status") != "OK":
        raise IOError(f"upload failed for {path}: {body}")
    df = body["data"]["files"][0]["dataFile"]
    remote_md5 = df.get("md5") or df.get("checksum", {}).get("value")
    local_md5 = _md5(path)
    if remote_md5 and remote_md5 != local_md5:
        raise IOError(f"checksum mismatch after upload for {path}: "
                      f"local {local_md5} != remote {remote_md5}")
    return df


def fetch(name, dest_dir, files=None, doi=DATASET_DOI, server=SERVER, overwrite=False):
    """Download one file by name into dest_dir and verify its MD5. Returns the local path."""
    files = files or list_files(doi, server)
    if name not in files:
        raise FileNotFoundError(f"{name} is not in {doi}")
    meta = files[name]
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, name)
    if os.path.exists(path) and not overwrite and (meta["md5"] is None or _md5(path) == meta["md5"]):
        return path
    url = f"{server}/api/access/datafile/{meta['id']}" + ("?format=original" if meta["original"] else "")
    tmp = path + ".part"
    with _get(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    if meta["md5"] is not None and _md5(tmp) != meta["md5"]:
        os.remove(tmp)
        raise IOError(f"checksum mismatch for {name}")
    os.replace(tmp, path)
    return path
