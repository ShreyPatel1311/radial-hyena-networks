"""
Is radial conditioning useful, and is radius the right field to condition on?
============================================================================

The RadialNodeStream does TWO separable things that the architecture conflates:

  ORDERING      which atom sits at which position in the sequence the Hyena long-conv
                runs along.  Nominally core -> surface by radius.
  CONDITIONING  the per-position feature the implicit filter is generated FROM.
                Nominally RBF(r / r_max), i.e. normalised radius.

Ablating "radial conditioning" as a single thing cannot separate the two. This script
sweeps them independently, plus a few interactions.

Eval-only: each number is a trained model evaluated under one variant.
"""
from __future__ import annotations
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from radial_hyena import data as D
from radial_hyena import model as A
from radial_hyena.config import DEFAULT
from radial_hyena.metrics import CLASSIFICATION_TASKS as CLS_TASKS, evaluate, metric_key
from radial_hyena.runner import build_model, load_task, make_loader, reproduction_gate
from sklearn.metrics import f1_score, accuracy_score

STATE = {"cond": "radial", "order": "as_trained", "deg": None, "gen": None}


@torch.no_grad()
def evaluate_safe(model, cpu_model, loader, task, dev):
    """Same metrics as kan_prune_test.evaluate, but with a per-batch CPU fallback.

    A 4 GB card cannot hold this model's edge-stream FFT buffers at the training batch
    size of 16, and the batch size CANNOT be lowered: the node stream's float32 ordering
    key makes activations depend on batch composition, so a smaller batch would silently
    change the very quantity being measured. Falling back to a persistent CPU copy keeps
    the maths identical.
    """
    model.eval()
    ys, ps, se, n = [], [], 0.0, 0
    for bt in loader:
        y = bt.y.clone()
        try:
            out = model(bt.to(dev)).float().cpu()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out = cpu_model(bt.to("cpu")).float()
        if task in CLS_TASKS:
            t = y.view(-1) if task != "atom" else y
            ps.append(out.argmax(-1)); ys.append(t)
        else:
            e = (out - y.reshape(out.shape)) ** 2
            se += float(e.sum()); n += y.numel()
    if task in CLS_TASKS:
        yv = torch.cat(ys).numpy(); pv = torch.cat(ps).numpy()
        keep = yv >= 0
        n_unmap = int((~keep).sum()); yv, pv = yv[keep], pv[keep]
        return {"weighted_f1": float(f1_score(yv, pv, average="weighted", zero_division=0)),
                "macro_f1": float(f1_score(yv, pv, average="macro", zero_division=0)),
                "acc": float(accuracy_score(yv, pv)),
                "n_eval": int(len(yv)), "n_unmappable": n_unmap}
    return {"mse": se / max(n, 1), "n_eval": int(n)}


def _ordering(mode, batch, r, N, dev):
    """Return a permutation that is ALWAYS grouped by graph (ascending), with the
    within-graph order set by `mode`."""
    if mode == "as_trained":
        return torch.argsort(batch.float() * 1e9 + r)          # as trained
    if mode == "input_order":
        key = r.new_zeros(N)                                   # constant -> stable = input order
    elif mode == "true_radial":
        key = r
    elif mode == "reverse_radial":
        key = -r
    elif mode == "random":
        key = torch.rand(N, device=dev, generator=STATE["gen"])
    else:
        raise ValueError(mode)
    idx = torch.argsort(key, stable=True)                      # within-graph order
    return idx[torch.argsort(batch[idx], stable=True)]         # then group by graph


def _conditioning(mode, rn, batch, N, dev, order, Nmax, offs, gs):
    if mode == "radial":
        return rn
    if mode == "constant":
        return torch.full_like(rn, 0.5)
    if mode == "reversed":
        return 1.0 - rn
    if mode == "random":
        return torch.rand(N, device=dev, generator=STATE["gen"])
    if mode == "shuffled":
        # permute rn WITHIN each graph: identical marginal distribution, but the pairing
        # between an atom's content and its own radius is destroyed.
        base = torch.argsort(batch, stable=True)               # atoms grouped by graph
        rnd = torch.argsort(torch.rand(N, device=dev, generator=STATE["gen"]), stable=True)
        perm = rnd[torch.argsort(batch[rnd], stable=True)]      # same grouping, shuffled inside
        out = rn.clone()
        out[base] = rn[perm]
        return out
    if mode == "position":
        # classic Hyena: condition on normalised sequence position, not on any physics
        rank = torch.arange(N, device=dev) - offs[gs]
        cnt = torch.bincount(batch, minlength=int(batch.max()) + 1).clamp(min=1)
        p = rank.float() / cnt[gs].float().clamp(min=1)
        out = rn.clone(); out[order] = p.clamp(0, 1)
        return out
    if mode == "coordination":
        deg = STATE["deg"]
        if deg is None:
            return rn
        dmax = torch.zeros(int(batch.max()) + 1, device=dev).scatter_reduce(
            0, batch, deg, reduce="amax", include_self=False).clamp(min=1e-6)
        return (deg / dmax[batch]).clamp(0, 1)
    raise ValueError(mode)


def patched_forward(self, h, pos, batch, B):
    if self.inject_pos:
        h = h + self.pos_embed(pos)
    N, d = h.shape; dev = h.device
    cen = A.scatter_mean(pos, batch, B)
    r = (pos - cen[batch]).norm(dim=-1)
    rmax = torch.zeros(B, device=dev).scatter_reduce(0, batch, r, reduce="amax",
                                                     include_self=False).clamp(min=1e-6)
    rn = (r / rmax[batch]).clamp(0, 1)

    counts = torch.bincount(batch, minlength=B); Nmax = int(counts.max().item())
    offs = F.pad(counts.cumsum(0), (1, 0))
    order = _ordering(STATE["order"], batch, r, N, dev)
    gs = batch[order]; rank = torch.arange(N, device=dev) - offs[gs]
    flat = gs * Nmax + rank

    rn_use = _conditioning(STATE["cond"], rn, batch, N, dev, order, Nmax, offs, gs)

    seq = h.new_zeros(B * Nmax, d); seq[flat] = h[order]
    cond = h.new_zeros(B * Nmax, self.num_rbf)
    cond[flat] = A.rbf_encode(rn_use[order], self.num_rbf, 0.0, 1.0)
    mask = torch.zeros(B * Nmax, dtype=torch.bool, device=dev); mask[flat] = True
    out = self.conv(seq.view(B, Nmax, d), cond.view(B, Nmax, self.num_rbf), mask.view(B, Nmax))
    h_out = h.new_zeros(N, d); h_out[order] = out.reshape(B * Nmax, d)[flat]
    return self.norm(h_out)


RUNS = [
    # (label, order_mode, cond_mode, what it isolates)
    ("baseline",              "as_trained",     "radial",       "as trained"),
    # --- ORDERING axis (conditioning held at the real radial field) ---
    ("order:true_radial",     "true_radial",    "radial",       "stable two-key radial sort"),
    ("order:reverse_radial",  "reverse_radial", "radial",       "surface->core"),
    ("order:random",          "random",         "radial",       "no ordering information"),
    ("order:input_order",     "input_order",    "radial",       "raw atom order from the file"),
    # --- CONDITIONING axis (ordering held as trained) ---
    ("cond:constant",         "as_trained",     "constant",     "no conditioning signal"),
    ("cond:shuffled",         "as_trained",     "shuffled",     "same marginal, wrong pairing"),
    ("cond:random",           "as_trained",     "random",       "noise conditioning"),
    ("cond:reversed",         "as_trained",     "reversed",     "core<->surface swapped"),
    ("cond:position",         "as_trained",     "position",     "classic Hyena positional"),
    ("cond:coordination",     "as_trained",     "coordination", "a DIFFERENT physical field"),
    # --- interactions ---
    ("fix+cond:constant",     "true_radial",    "constant",     "fixed order, no conditioning"),
    ("fix+cond:position",     "true_radial",    "position",     "fixed order, positional cond"),
    ("fix+cond:coordination", "true_radial",    "coordination", "fixed order, coordination cond"),
    # --- conditioning measured in the regime the model ACTUALLY operates in ---
    # On CUDA, argsort over the fully-tied collapsed key returns the IDENTITY permutation,
    # so the trained model saw INPUT-ORDER sequences, never radial ones. Ablating the
    # conditioning against the as-trained CPU ordering therefore measures nothing useful;
    # it has to be ablated against input order.
    ("inp+cond:radial",       "input_order",    "radial",       "TRUE OPERATING POINT"),
    ("inp+cond:constant",     "input_order",    "constant",     "no conditioning, real regime"),
    ("inp+cond:shuffled",     "input_order",    "shuffled",     "wrong pairing, real regime"),
    ("inp+cond:random",       "input_order",    "random",       "noise cond, real regime"),
    ("inp+cond:position",     "input_order",    "position",     "positional cond, real regime"),
    ("inp+cond:coordination", "input_order",    "coordination", "coordination cond, real regime"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(D.TASKS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--data-zip", default=None)
    ap.add_argument("--out-dir", default="results/conditioning")
    ap.add_argument("--batch-size", type=int, default=DEFAULT.batch_size)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeats for the stochastic variants")
    args = ap.parse_args()
    TASK, dev = args.task, args.device
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    sub, splits, order, sg, records, out_dim = load_task(TASK, args.cache_dir, args.data_zip)
    loader = make_loader(records, order, TASK, sg, args.batch_size)
    model, ck = build_model(TASK, out_dim, args.checkpoint, dev)
    key = metric_key(TASK)

    def stash_degree(mod, inp):
        d = inp[0]
        src = d.edge_index[0]
        src = src[src != d.edge_index[1]]
        STATE["deg"] = torch.bincount(src, minlength=d.x.shape[0]).float()
    model.register_forward_pre_hook(stash_degree)

    orig = A.RadialNodeStream.forward
    A.RadialNodeStream.forward = patched_forward
    STATE["gen"] = torch.Generator(device=dev)

    # GATE: the patched path with the as-trained settings must reproduce the checkpoint
    STATE.update(cond="radial", order="as_trained")
    STATE["gen"].manual_seed(0)
    base = evaluate_safe(model, cpu_model, loader, TASK, dev)
    ref = ck["test"][key]
    ok = abs(base[key] - ref) <= 2e-3
    print(f"[{TASK}] patched-path baseline {key}={base[key]:.6f} vs checkpoint {ref:.6f} "
          f"-> {'REPRODUCED' if ok else 'MISMATCH'}", flush=True)

    STOCH = {"random", "shuffled"}
    results = {}
    for label, omode, cmode, what in RUNS:
        n_rep = args.repeats if (omode in STOCH or cmode in STOCH) else 1
        vals, ms = [], None
        for rep in range(n_rep):
            STATE.update(cond=cmode, order=omode)
            STATE["gen"].manual_seed(1234 + rep)
            m = evaluate_safe(model, cpu_model, loader, TASK, dev)
            vals.append(m[key]); ms = ms or m
        results[label] = {"order": omode, "cond": cmode, "isolates": what,
                          "metric": float(np.mean(vals)),
                          "sd": float(np.std(vals)) if n_rep > 1 else 0.0,
                          "n_repeats": n_rep, "full": ms}
        rel = 100 * (np.mean(vals) - base[key]) / abs(base[key])
        print(f"  {label:24s} {key}={np.mean(vals):.5f}"
              f"{'±' + format(np.std(vals), '.5f') if n_rep > 1 else '        '}"
              f"  ({rel:+6.1f}% vs baseline)   [{what}]", flush=True)

    A.RadialNodeStream.forward = orig
    out = {"task": TASK, "device": dev, "metric_key": key, "baseline_reproduced": bool(ok),
           "baseline": base, "checkpoint_test": ck["test"], "n_test_graphs": len(eval_idx),
           "n_test_graphs_full": len(te), "max_atoms": args.max_atoms, "device": args.device,
           "batch_size": args.batch_size, "runs": results, "elapsed_sec": time.time() - t0}
    p = os.path.join(args.out_dir, f"conditioning_{TASK}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"[{TASK}] saved -> {p} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
