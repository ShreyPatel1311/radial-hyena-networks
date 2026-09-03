"""
Descriptor couplings over the FULL benchmark subset (2,975 particles), by split.

The published Fig-2 couplings used the 298 test particles only. With 64 units x 14
descriptors = 896 tests per task, n=298 puts the multiple-comparison noise ceiling at
~0.23, so mid-range cells were not individually interpretable. This recomputes everything
on all 2,975 graphs.

Merging train into the analysis is NOT automatically safe: the model reaches train
wF1 = 1.0 by ~epoch 35, so its internal representation of a memorised training particle
may differ systematically from a held-out one. This script therefore computes couplings
per split FIRST and reports train-vs-test agreement; the merged heatmap is only
interpretable if they agree.
"""
from __future__ import annotations
import os, sys, json, argparse

import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena.couplings import partial_spearman, CONFOUND, SKIP_DESC

TASKS = ["crystal_system", "space_group", "atom", "saxs", "xrd", "xpdf"]
NICE = {"crystal_system": "Crystal system", "space_group": "Space group",
        "atom": "Atom type", "saxs": "SAXS", "xrd": "XRD", "xpdf": "xPDF"}


def coupling_matrix(X, Dm, Z, rows):
    """|partial Spearman| for every (unit, descriptor) pair over the rows given."""
    C = np.zeros((X.shape[1], Dm.shape[1]))
    for k in range(Dm.shape[1]):
        d = Dm[rows, k]
        m = np.isfinite(d)
        if m.sum() < 20:
            continue
        dv, Zv = d[m], Z[rows][m]
        for j in range(X.shape[1]):
            C[j, k] = abs(partial_spearman(X[rows][m, j], dv, Zv))
    return C


def null_ceiling(n_total, Dm, Z, rows, n_units=64, draws=40, seed=0):
    """Largest |partial Spearman| over the whole 64 x n_desc grid under pure noise.

    X must span the FULL row space, not just len(rows): coupling_matrix indexes it with
    absolute row indices.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        X = rng.normal(size=(n_total, n_units))
        C = coupling_matrix(X, Dm, Z, rows)
        out.append(C.max())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="results/activations_all.npz")
    ap.add_argument("--desc", default="results/physics_descriptors_all.npz")
    ap.add_argument("--out", default="results/couplings_all_splits.npz")
    ap.add_argument("--null-draws", type=int, default=25)
    a = ap.parse_args()

    A_ = np.load(a.acts, allow_pickle=True)
    Dz = np.load(a.desc, allow_pickle=True)
    # descriptor file and activation file are both ordered by sorted(index) -- verify
    assert np.array_equal(A_["index"], Dz["index"]), "activation/descriptor row mismatch"
    split = np.array([str(s) for s in A_["split"]])
    nm = [str(x) for x in Dz["names"]]
    V = Dz["values"]
    names = [x for x in nm if x not in SKIP_DESC]
    Dm = V[:, [nm.index(x) for x in names]]
    Z = V[:, [nm.index(c) for c in CONFOUND]]
    N = len(split)
    idx = {"train": np.where(split == "train")[0], "val": np.where(split == "val")[0],
           "test": np.where(split == "test")[0], "merged": np.arange(N)}
    print(f"{N} particles | " + " ".join(f"{k}={len(v)}" for k, v in idx.items()))
    print(f"{len(names)} descriptors x 64 units = {64*len(names)} tests per task\n")

    store = {"names": np.array(names)}
    summary = {}
    for t in TASKS:
        key = f"{t}_b1"
        if key not in A_:
            print(f"  [skip] {t}: no activations"); continue
        X = A_[key]
        C = {s: coupling_matrix(X, Dm, Z, idx[s]) for s in idx}
        for s, M in C.items():
            store[f"{t}_{s}"] = M
        # train-vs-test agreement over the whole 896-cell grid
        ag = spearmanr(C["train"].ravel(), C["test"].ravel()).statistic
        top_tr = names[int(np.argmax(C["train"].max(0)))]
        top_te = names[int(np.argmax(C["test"].max(0)))]
        top_mg = names[int(np.argmax(C["merged"].max(0)))]
        summary[t] = {"agreement_train_test": float(ag),
                      "top_train": top_tr, "top_test": top_te, "top_merged": top_mg,
                      "peak_train": float(C["train"].max()),
                      "peak_test": float(C["test"].max()),
                      "peak_merged": float(C["merged"].max()),
                      "units_above_0.4_merged": int((C["merged"].max(1) > 0.4).sum())}
        print(f"{NICE[t]:16s} train-vs-test grid agreement rho={ag:+.3f} | "
              f"top: train={top_tr}, test={top_te}, merged={top_mg} | "
              f"peak merged={C['merged'].max():.3f}")

    print("\nmultiple-comparison ceiling (max over the grid under pure noise):")
    for s in ["test", "merged"]:
        mx = null_ceiling(N, Dm, Z, idx[s], draws=a.null_draws)
        summary[f"null_{s}"] = {"mean": float(mx.mean()), "p95": float(np.percentile(mx, 95)),
                                "n": int(len(idx[s]))}
        print(f"  {s:7s} (n={len(idx[s]):4d}): mean {mx.mean():.3f}, 95th pct "
              f"{np.percentile(mx,95):.3f}")
        store[f"null_{s}"] = mx

    np.savez_compressed(a.out, **store)
    json.dump(summary, open(a.out.replace(".npz", ".json"), "w"), indent=2)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
