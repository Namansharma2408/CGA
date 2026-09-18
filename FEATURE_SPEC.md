# FEATURE_SPEC — Canonical Feature Definitions (P0-2/P0-4)

Single source of truth for feature indices, formulas, and dtypes.
Python (`scripts/`, `improved/scripts/`) and C++ (`src/models/`,
`improved/models/`) MUST emit identical order/names. Training MUST fail if
`clf.n_features_in_ != DIM`.

## 19-dim (basic, Table II: M×9, V×5, H×5)

| Idx | Name | Formula | Dtype |
|-----|------|---------|-------|
| 0 | log_n | log1p(n) | float32 |
| 1 | log_nnz | log1p(nnz) | float32 |
| 2 | log_avg_degree | log1p(nnz/n) | float32 |
| 3 | row_cv | row_std/row_mean (0 if mean==0) | float32 |
| 4 | col_cv | col_std/col_mean from **CSC** indptr (P0-3) | float32 |
| 5 | log_max_row_nnz | log1p(row_max) | float32 |
| 6 | gini_row | gini(row_counts) | float32 |
| 7 | gini_col | gini(col_counts) | float32 |
| 8 | is_scale_free | 1 if gini_row>0.4 or avg_degree>10 else 0 | 0/1 |
| 9 | log_x_nnz | log1p(x_nnz) | float32 |
| 10 | x_density | x_nnz/n | float32 |
| 11 | log_m_nnz | log1p(m_nnz) | float32 |
| 12 | m_density | **m_nnz/nnz** (P0-2; was m_nnz/n in C++) | float32 |
| 13 | x_decreasing | 1 if x_nnz<x_prev and x_prev>0 else 0 | 0/1 |
| 14 | log_nnz_ratio | log1p(x_nnz/m_nnz) | float32 |
| 15 | log_valid_nnz | log1p(nnz*xd+1) | float32 |
| 16 | degree_x | avg_degree*xd | float32 |
| 17 | degree_x_m | avg_degree*xd*md | float32 |
| 18 | log_masked_valid_nnz | log1p(nnz*xd*md+1) | float32 |

`compute_matrix_stats` MUST take both CSR and CSC indptrs; `col_*` from CSC.
`m_nnz` remaining-nnz proxy: `max(0, m_nnz - x_nnz*avg_degree)` (heuristic,
documented; keeps md in [0,1]).

## 25-dim (improved = 19 above + 6 new)

| Idx | Name | Formula |
|-----|------|---------|
| 19 | degree_variance | row_std² |
| 20 | hash_pressure | min(1024, x_nnz/16+1)/1024 |
| 21 | sparsity_aware_gini | gini_row*(1-xd) |
| 22 | load_balance_factor | avg_degree*(0.5+0.5*md) |
| 23 | growth_rate | log1p(x_nnz/(x_prev+1)) (1.0 if x_prev==0) |
| 24 | x_m_density | xd*md |

Notes:
- Index 4 is ALWAYS col_cv (P0-4 fix; degree_variance moved to 19).
- `xfer_cost_us` (was constant 0.15, zero variance) REMOVED from trained vector.
  Use `BANDWIDTH_GBS=12.0` separately for cost modelling if needed.
- DTA `t_var` threshold reads index 19, not 4.
