# Reproducing the reported results

All commands assume the cache has been built:

```bash
python scripts/prepare_data.py --data-zip /path/to/CHILI-100K.zip
```

## 1. Benchmark table

```bash
make train
make evaluate
```

Parameter counts, asserted on checkpoint load and by `tests/test_model.py`:

| task | out_dim | params |
|---|---|---|
| crystal_system | 7 | 220,878 |
| space_group | 151 | 303,822 |
| atom | 118 | 284,808 |
| saxs | 300 | 389,646 |
| xrd | 580 | 550,926 |
| xpdf | 6000 | 3,672,846 |

## 2. KAN spline pruning

```bash
make pruning
```

Writes `results/kan_pruning/kan_pruning_<task>.json`:

* `axis1_edge_pruning` — metric vs fraction of spline edges removed, under three orderings
* `axis2_grid_coarsening` — metric vs spline basis size (8 to 4 coefficients per edge)
* `reference_points` — `spline_off` (base path only) and `base_off` (splines only)

## 3. Conditioning ablation

```bash
make conditioning
```

Writes `results/conditioning/conditioning_<task>.json`. Twenty variants sweep two axes:

* ordering — `as_trained`, `input_order`, `true_radial`, `reverse_radial`, `random`
* conditioning — `radial`, `constant`, `shuffled`, `random`, `reversed`, `position`,
  `coordination`

Rows prefixed `inp+` hold the ordering fixed while varying the conditioning.

## 4. Spline–physics analysis

```bash
make physics
```

1. `physical_axes.py` — q (SAXS/XRD) and r (xPDF) grids
2. `physics_descriptors.py` — 20 descriptors per particle from coordinates
3. `spline_operator.py` — per-branch read-out Jacobians and spline nonlinearity
4. `spline_physics.py` — projects Jacobian columns onto `sinc(qr)` with permutation nulls

## 5. Descriptor couplings

```bash
python experiments/descriptor_couplings.py
```

Per-unit partial Spearman couplings to each descriptor, computed per split and merged,
with the multiple-comparison ceiling from permutation.
