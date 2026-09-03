"""
Does the KAN read-out implement the physics that maps structure -> target?
=========================================================================

Consumes the raw operator artifacts (src/mech_spline_operator.py, run on GPU) and the
physical descriptors computed from the real coordinates (src/physics_descriptors.py).
All analysis here is pure numpy so it can be iterated without a GPU.

Three tests, each with an explicit null:

TEST 1 -- Debye projection (SAXS, XRD).
  The Debye equation, I(q) = sum_ij f_i f_j sinc(q r_ij), says any physically-faithful
  read-out over the q axis must be a superposition of sinc(q r) kernels, one per
  interatomic distance r. So project each hidden unit's read-out Jacobian column J[:, j]
  -- a function of q -- onto the kernel bank K[q, r] = sinc(q r). The resulting
  correlation spectrum rho_j(r) is that unit's implied DISTANCE CONTENT. Aggregated over
  units (weighted by how much each unit actually varies across the test set) it gives the
  model's implied distance spectrum S(r), which is then compared against the TRUE mean
  pair-distance distribution P(r) measured from the coordinates.

TEST 2 -- xPDF locality.
  For xPDF the output index IS r, so no projection is needed: J[:, j] is already a
  function of distance. Physics says G(r) is a sum of sharp coordination shells, so the
  read-out's aggregate |J|(r) profile should track the true P(r).

TEST 3 -- descriptor coupling vs spline nonlinearity (all six tasks).
  For each of the 64 hidden units, measure (a) its Spearman coupling to each physical
  descriptor (R_g, Q4/Q6/Q8, coordination, first-shell distance, ...) and (b) how
  NONLINEAR the splines reading that unit are. If the splines exist to capture physics
  the linear branch cannot, units carrying physical descriptors should be read out
  through more nonlinear splines than units that do not.

Nulls throughout: per-column row shuffles and circular shifts of J, which preserve every
marginal (magnitudes, per-unit variance) and destroy only the ordering along the physical
axis -- the exact structure the physics predicts.
"""
from __future__ import annotations
import os, sys, json, glob, argparse
import numpy as np
from scipy.stats import spearmanr


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radial_hyena.couplings import couplings as partial_couplings

AXES = {"saxs": "SAXS", "xrd": "XRD", "xpdf": "xPDF"}
CLS = ("crystal_system", "space_group", "atom")
R_GRID = np.arange(1.0, 30.01, 0.1)          # distances probed, Angstrom
N_NULL = 200


def sinc_kernel(q, r_grid):
    """K[q, r] = sin(q r)/(q r), the Debye kernel; the q->0 limit is 1."""
    qr = np.outer(q, r_grid)
    K = np.where(qr > 1e-12, np.sin(np.clip(qr, 1e-12, None)) / np.where(qr > 1e-12, qr, 1), 1.0)
    return K


def _zc(a, axis=0):
    a = a - a.mean(axis=axis, keepdims=True)
    s = a.std(axis=axis, keepdims=True)
    return a / np.where(s > 1e-12, s, 1.0)


def corr_spectrum(J, K):
    """rho[r, j] = Pearson corr over the physical axis between J[:, j] and K[:, r]."""
    Jz, Kz = _zc(J, 0), _zc(K, 0)
    return (Kz.T @ Jz) / J.shape[0]


def aggregate_spectrum(J, K, w):
    rho = corr_spectrum(J, K)                       # (r, j)
    S = np.abs(rho) @ w
    return S / max(S.sum(), 1e-12)


def null_spectra(J, K, w, rng, n=N_NULL):
    out = []
    for _ in range(n):
        Jn = np.empty_like(J)
        for j in range(J.shape[1]):
            if rng.random() < 0.5:
                Jn[:, j] = rng.permutation(J[:, j])                 # row shuffle
            else:
                Jn[:, j] = np.roll(J[:, j], rng.integers(1, J.shape[0]))   # circular shift
        out.append(aggregate_spectrum(Jn, K, w))
    return np.array(out)


def match_score(S, P):
    """Spearman between the model's implied distance spectrum and the true P(r)."""
    m = np.isfinite(S) & np.isfinite(P)
    if m.sum() < 5:
        return np.nan
    return float(spearmanr(S[m], P[m]).statistic)


def true_pair_distribution(desc, r_grid):
    P = desc["pair_hist"].mean(0)                    # mean over test structures
    rc = desc["r_centers"]
    return np.interp(r_grid, rc, P)


def unit_weights(x, gids, n_graphs):
    """How much each hidden unit actually varies across TEST STRUCTURES (node-level tasks
    are averaged within a graph first, so a unit that only varies within a particle does
    not masquerade as structure-discriminating)."""
    if x.shape[0] != n_graphs:
        agg = np.zeros((n_graphs, x.shape[1]))
        cnt = np.zeros(n_graphs)
        np.add.at(agg, gids, x)
        np.add.at(cnt, gids, 1)
        x = agg / np.maximum(cnt, 1)[:, None]
    w = x.std(0)
    return w / max(w.sum(), 1e-12), x


def analyse_task(op, desc, rng):
    task = str(op["task"])
    n_graphs = int(op["n_test_graphs"])
    names = list(desc["names"])
    V = desc["values"]
    res = {"task": task}

    Jb, Js = op["b1_J_base"], op["b1_J_spline"]
    Jt = Jb + Js
    w, hmat = unit_weights(op["b1_x"], op["graph_ids"], n_graphs)

    # ---------------- TEST 1 / 2: operator vs physics along the physical axis ------------
    if task in AXES:
        ax = np.load(a.axes)[AXES[task]]
        P_true = true_pair_distribution(desc, R_GRID)
        if task == "xpdf":
            # output axis already IS r: read the aggregate |J| profile on the model's own
            # grid and resample onto R_GRID for comparison.
            K = None
            prof = {}
            for nm, J in (("total", Jt), ("base", Jb), ("spline", Js)):
                p = np.abs(J) @ w
                prof[nm] = np.interp(R_GRID, ax, p)
                prof[nm] = prof[nm] / max(prof[nm].sum(), 1e-12)
            S = prof
            nulls = {}
            for nm, J in (("total", Jt), ("base", Jb), ("spline", Js)):
                ns = []
                for _ in range(N_NULL):
                    Jn = np.empty_like(J)
                    for j in range(J.shape[1]):
                        Jn[:, j] = (rng.permutation(J[:, j]) if rng.random() < 0.5
                                    else np.roll(J[:, j], rng.integers(1, J.shape[0])))
                    p = np.interp(R_GRID, ax, np.abs(Jn) @ w)
                    ns.append(p / max(p.sum(), 1e-12))
                nulls[nm] = np.array(ns)
        else:
            K = sinc_kernel(ax, R_GRID)
            S = {nm: aggregate_spectrum(J, K, w)
                 for nm, J in (("total", Jt), ("base", Jb), ("spline", Js))}
            nulls = {nm: null_spectra(J, K, w, rng)
                     for nm, J in (("total", Jt), ("base", Jb), ("spline", Js))}

        res["physics_axis"] = {"name": "r (A)" if task == "xpdf" else "q (1/A)",
                               "n": int(ax.size), "lo": float(ax[0]), "hi": float(ax[-1])}
        res["distance_spectrum"] = {}
        for nm in S:
            obs = match_score(S[nm], P_true)
            nd = np.array([match_score(n, P_true) for n in nulls[nm]])
            nd = nd[np.isfinite(nd)]
            z = (obs - nd.mean()) / max(nd.std(), 1e-12) if nd.size else np.nan
            res["distance_spectrum"][nm] = {
                "spearman_vs_true_Pr": obs,
                "null_mean": float(nd.mean()) if nd.size else None,
                "null_sd": float(nd.std()) if nd.size else None,
                "z": float(z),
                "p_one_sided": float((np.sum(nd >= obs) + 1) / (nd.size + 1)) if nd.size else None,
                "peak_r_A": float(R_GRID[int(np.argmax(S[nm]))]),
            }
        res["_spectra"] = {k: v.tolist() for k, v in S.items()}
        res["_true_Pr"] = P_true.tolist()
        res["_r_grid"] = R_GRID.tolist()

    # ---------------- TEST 3: descriptor coupling vs spline nonlinearity ----------------
    NL = op["b1_NL"]                                    # (out, in) nonlinearity per spline
    mag = np.abs(Js)
    nl_unit = (NL * mag).sum(0) / np.maximum(mag.sum(0), 1e-12)   # magnitude-weighted
    lin_share = np.abs(Jb).sum(0) / np.maximum(np.abs(Jb).sum(0) + np.abs(Js).sum(0), 1e-12)

    # Couplings are PARTIAL: the model is handed particle size directly in graph_attr, and
    # R_g correlates 0.98 with size_rank, so a raw correlation would report a unit echoing
    # its own input as recovered Guinier physics. See radial_hyena.couplings.
    Craw, dnames, _ = partial_couplings(op, desc, "b1", partial=False)
    Cpar, _, _ = partial_couplings(op, desc, "b1", partial=True)
    coup = {dn: {"max_abs_rho_partial": float(Cpar[:, k].max()),
                 "max_abs_rho_raw": float(Craw[:, k].max()),
                 "argmax_unit": int(np.argmax(Cpar[:, k])),
                 "mean_abs_rho_partial": float(Cpar[:, k].mean())}
            for k, dn in enumerate(dnames)}
    best = Cpar.max(1)
    res["descriptor_coupling"] = coup
    res["unit_summary"] = {
        "spline_nonlinearity_mean": float(nl_unit.mean()),
        "base_share_of_jacobian_mean": float(lin_share.mean()),
    }

    # do units that encode physics get read out through MORE nonlinear splines?
    rho_nl = spearmanr(best, nl_unit)
    perm = np.array([spearmanr(rng.permutation(best), nl_unit).statistic for _ in range(2000)])
    res["nonlinearity_vs_physics_coupling"] = {
        "spearman": float(rho_nl.statistic),
        "p_analytic": float(rho_nl.pvalue),
        "p_permutation": float((np.sum(np.abs(perm) >= abs(rho_nl.statistic)) + 1) / 2001),
        "n_units": int(best.size),
    }
    res["_best_coupling"] = best.tolist()
    res["_nl_unit"] = nl_unit.tolist()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", default="results/spline_operator")
    ap.add_argument("--descriptors", default="results/physics_descriptors_test.npz")
    ap.add_argument("--axes", default="results/physical_axes.npz")
    ap.add_argument("--out", default="results/spline_physics.json")
    a = ap.parse_args()
    desc = np.load(a.descriptors, allow_pickle=True)
    rng = np.random.default_rng(0)
    allres = []
    for p in sorted(glob.glob(os.path.join(a.op_dir, "operator_*.npz"))):
        op = np.load(p, allow_pickle=True)
        r = analyse_task(op, desc, rng)
        allres.append(r)
        print(f"\n=== {r['task']} ===")
        if "distance_spectrum" in r:
            for nm, v in r["distance_spectrum"].items():
                print(f"  {nm:7s} spearman(S(r), true P(r)) = {v['spearman_vs_true_Pr']:+.3f} "
                      f"| null {v['null_mean']:+.3f}+-{v['null_sd']:.3f} | z={v['z']:+.1f} "
                      f"| p={v['p_one_sided']:.4f} | peak r={v['peak_r_A']:.1f} A")
        n = r["nonlinearity_vs_physics_coupling"]
        print(f"  nonlinearity vs physics coupling: rho={n['spearman']:+.3f} "
              f"(perm p={n['p_permutation']:.4f}, n={n['n_units']})")
        top = sorted(r["descriptor_coupling"].items(),
                     key=lambda kv: -kv[1]["max_abs_rho_partial"])[:6]
        print("  strongest PARTIAL descriptor couplings: " +
              ", ".join(f"{k}={v['max_abs_rho_partial']:.2f}(raw {v['max_abs_rho_raw']:.2f})"
                        for k, v in top))
    json.dump(allres, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
