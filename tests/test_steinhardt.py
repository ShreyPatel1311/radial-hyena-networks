"""Steinhardt order parameters against ideal-lattice constants.

bcc is conventionally defined on its 14-neighbour shell; with an 8-neighbour cutoff the
correct values are Q4 = 0.509, Q6 = 0.629.
"""
import itertools
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from radial_hyena.physics import steinhardt

LITERATURE = {                      # (cutoff in lattice units, Q4, Q6)
    "fcc": (0.75, 0.191, 0.575),
    "hcp": (1.05, 0.097, 0.485),
    "sc":  (1.10, 0.764, 0.354),
    "bcc": (1.10, 0.036, 0.511),
}


def _lattice(kind, n=4):
    basis = {"sc": [(0, 0, 0)],
             "bcc": [(0, 0, 0), (.5, .5, .5)],
             "fcc": [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]}[kind]
    return np.array([np.array(c, float) + np.array(b)
                     for c in itertools.product(range(-n, n + 1), repeat=3) for b in basis])


def _hcp(n=4):
    ca = np.sqrt(8 / 3)
    a1, a2, a3 = np.array([1, 0, 0]), np.array([.5, np.sqrt(3) / 2, 0]), np.array([0, 0, ca])
    basis = [np.zeros(3), np.array([.5, np.sqrt(3) / 6, ca / 2])]
    return np.array([i * a1 + j * a2 + k * a3 + b
                     for i, j, k in itertools.product(range(-n, n + 1), repeat=3)
                     for b in basis])


@pytest.mark.parametrize("kind", list(LITERATURE))
def test_reproduces_ideal_lattice_constants(kind):
    cutoff, q4_ref, q6_ref = LITERATURE[kind]
    pos = _hcp() if kind == "hcp" else _lattice(kind)
    pos = pos[np.linalg.norm(pos - pos.mean(0), axis=-1) < 3.2]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    src, dst = np.where((d > 1e-6) & (d < cutoff))
    ei = np.stack([src, dst])
    deg = np.bincount(ei[0], minlength=len(pos))
    interior = np.isin(ei[0], np.where(deg == deg.max())[0])
    q = steinhardt(pos, ei[:, interior], cap=600)
    assert q["Q4"] == pytest.approx(q4_ref, abs=2e-3)
    assert q["Q6"] == pytest.approx(q6_ref, abs=2e-3)
