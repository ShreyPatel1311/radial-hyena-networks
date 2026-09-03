"""The split must match the benchmark protocol."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from radial_hyena import data as D

CACHE = os.environ.get("RH_CACHE", "cache")
INDEX = os.path.join(CACHE, "chili_index.pkl")
needs_index = pytest.mark.skipif(
    not os.path.exists(INDEX),
    reason="run scripts/prepare_data.py first to build the index")


@needs_index
def test_split_matches_benchmark_protocol():
    index = D.build_index(None, cache=INDEX)
    sub = D.benchmark_subset(index)
    tr, va, te = D.make_split(sub)
    voc = D.space_group_vocab(sub, tr)
    unmappable = sum(1 for i in te if int(sub[i][3]) not in voc)
    assert len(sub) == 2975
    assert (len(tr), len(va), len(te)) == (2379, 298, 298)
    assert len(voc) == 151
    assert unmappable == 3


@needs_index
def test_split_is_deterministic():
    index = D.build_index(None, cache=INDEX)
    assert D.make_split(D.benchmark_subset(index)) == D.make_split(D.benchmark_subset(index))
