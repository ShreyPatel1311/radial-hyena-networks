import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from radial_hyena import hub                            # noqa: E402


def _md5(b):
    return hashlib.md5(b).hexdigest()


def test_parse_manifest_handles_md5sum_formats():
    text = ("d41d8cd98f00b204e9800998ecf8427e  ./data/chili100k_benchmark.h5\n"
            "\n"
            "0cc175b9c0f1b6a831c399e269772661 *checkpoints/atom_seed0.pt\n")
    m = hub.parse_manifest(text)
    assert m == {"data/chili100k_benchmark.h5": "d41d8cd98f00b204e9800998ecf8427e",
                 "checkpoints/atom_seed0.pt": "0cc175b9c0f1b6a831c399e269772661"}


def test_verify_local_reports_ok_missing_and_mismatch(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "data" / "a.h5").write_bytes(b"good")
    (tmp_path / "checkpoints" / "b.pt").write_bytes(b"tampered")
    (tmp_path / hub.MANIFEST).write_text(
        f"{_md5(b'good')}  ./data/a.h5\n"
        f"{_md5(b'original')}  ./checkpoints/b.pt\n"
        f"{_md5(b'x')}  ./records/c.json\n")
    assert hub.verify_local(str(tmp_path)) == {"data/a.h5": "ok",
                                              "checkpoints/b.pt": "mismatch",
                                              "records/c.json": "missing"}
