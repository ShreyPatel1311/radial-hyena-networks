"""
Shared helpers for the spline<->physics analysis.

The important one is `couplings(..., partial=True)`.

The model is HANDED particle size: graph_attr = [log1p(n_atoms), n_species, size_rank].
Because R_g correlates 0.98 with size_rank and 0.96 with log(n_atoms), a hidden unit that
merely echoes its own input scores |rho| ~ 0.98 against R_g and looks like recovered
Guinier physics when it is nothing of the sort. Every coupling reported must therefore be
PARTIAL: both the unit activation and the descriptor are residualised against the three
graph attributes first, so what remains is structure the model was not simply told.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, rankdata

R_GRID = np.arange(1.0, 30.01, 0.1)
SCAT = {"saxs": "SAXS", "xrd": "XRD", "xpdf": "xPDF"}
# descriptors that ARE the model's own graph-attribute inputs, or are labels
CONFOUND = ["log_n_atoms", "n_species", "size_rank"]
SKIP_DESC = {"crystal_system", "space_group", "size_rank", "n_atoms", "log_n_atoms",
             "n_species"}


def sinc_kernel(q, r):
    qr = np.outer(q, r)
    return np.where(qr > 1e-12, np.sin(np.clip(qr, 1e-12, None)) / np.where(qr > 1e-12, qr, 1), 1.0)


def zc(a):
    a = a - a.mean(0, keepdims=True)
    s = a.std(0, keepdims=True)
    return a / np.where(s > 1e-12, s, 1.0)


def per_graph(x, gids, n_graphs):
    if x.shape[0] == n_graphs:
        return x
    agg = np.zeros((n_graphs, x.shape[1])); cnt = np.zeros(n_graphs)
    np.add.at(agg, gids, x); np.add.at(cnt, gids, 1)
    return agg / np.maximum(cnt, 1)[:, None]


def _rank(a):
    r = rankdata(a).astype(float)
    return (r - r.mean()) / max(r.std(), 1e-12)


def _resid(y, Z):
    """Residual of y after least-squares removal of the confound columns Z (+ intercept)."""
    A = np.column_stack([np.ones(len(y)), zc(Z)])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ b


def partial_spearman(x, y, Z):
    """Partial SPEARMAN: rank-transform everything FIRST, then residualise.

    Residualising the raw values instead is not rank-preserving, so it gives different
    answers for variables that are monotone reparametrisations of each other. That bit us:
    surface_to_volume = 3/R_max has Spearman -1.000 with R_max, yet value-residualised
    partials came out 0.78 vs 0.41 -- an artefact of the choice of parametrisation, not a
    fact about the model. Ranking first makes the measure invariant to any monotone
    recoding of a descriptor, which is the whole point of using Spearman.
    """
    xr, yr = _rank(x), _rank(y)
    Zr = np.column_stack([_rank(Z[:, k]) for k in range(Z.shape[1])])
    rx, ry = _resid(xr, Zr), _resid(yr, Zr)
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.mean((rx / sx) * (ry / sy)))


def unit_r_star(op, axes, task):
    J = op["b1_J_base"] + op["b1_J_spline"]
    if task == "xpdf":
        r = axes["xPDF"]
        return r[np.argmax(np.abs(J), axis=0)], np.abs(J).max(0) / max(np.abs(J).max(), 1e-12)
    K = sinc_kernel(axes[SCAT[task]], R_GRID)
    rho = (zc(K).T @ zc(J)) / J.shape[0]
    return R_GRID[np.argmax(np.abs(rho), axis=0)], np.abs(rho).max(0)


def couplings(op, desc, block="b1", partial=True):
    """|Spearman| (partial by default) between each unit and each physical descriptor."""
    n = int(op["n_test_graphs"])
    X = per_graph(op[f"{block}_x"], op["graph_ids"], n)
    allnames = list(desc["names"])
    names = [d for d in allnames if d not in SKIP_DESC]
    V = desc["values"][:, [allnames.index(d) for d in names]]
    Z = desc["values"][:, [allnames.index(c) for c in CONFOUND]]
    C = np.zeros((X.shape[1], len(names)))
    for k in range(len(names)):
        d = V[:, k]
        m = np.isfinite(d)
        dv = d[m]
        for j in range(X.shape[1]):
            xv = X[m, j]
            if xv.std() < 1e-12:
                continue
            C[j, k] = (np.nan_to_num(partial_spearman(xv, d[m], Z[m])) if partial
                       else np.nan_to_num(spearmanr(xv, dv).statistic))
    return np.abs(C), names, X
