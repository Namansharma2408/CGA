# CGA Improved — Change Log & Build Guide
**Group Members:** Aditya Jha, Lakshay Gupta, Hirannya Mhaisbadwe, Naman Sharma

## What Changed

### `improved/` folder layout

```
improved/
├── CMakeLists.txt              ← build system (links to src/ for unchanged files)
├── README.md                   ← this file
├── models/
│   ├── features.h              ← DIM 19→25, fixed compute_matrix_stats signature
│   ├── features.cpp            ← col_counts fix + F1-F6 (indices 19-24)
│   ├── inference.h             ← KernelStats + SW-UCB fields + t_var threshold
│   └── inference.cpp           ← all 6 DTA improvements + Model-0 + one-copy fix
├── core/
│   └── bfs.cpp                 ← passes A_csc.indptr to compute_matrix_stats()
└── scripts/
    ├── feature_extraction.py   ← 25-dim extractor
    ├── train_models.py         ← improved training (25-dim, Eq.6 weights, depths)
    └── collect_training_data.py← ground-truth label collector (needs built binary)
```

---

## Bug Fixes Applied

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `features.cpp:33` | `col_counts = row_counts` (wrong for directed) | CSC `indptr` passed in |
| 2 | `features.h:35` | signature takes only CSR `indptr` | Added `csc_indptr` param |
| 3 | `bfs.cpp:49` | Missing `A_csc.indptr` in stats call | Both indptrs passed |
| 4 | `inference.cpp:53` | UCB bonus shrinks to ~0 after 20 iters | SW-UCB sliding window |
| 5 | `inference.cpp:68` | `explore_left` never decrements | Forced probe every 10 iters |
| 6 | `inference.cpp:122` | DTA bypasses Model1 + one-copy check | Both now active in DTA mode |
| 7 | `train_models.py` | Random features + heuristic labels | Real stats + argmin timing |

---

## New Features (F1–F6, indices 19-24)

| Idx | Name | Formula | Why |
|---|---|---|---|
| 19 | `degree_var` | `row_std²` | Hub dominance beyond Gini |
| 20 | `growth_rate` | `log(x_nnz / max(x_nnz_prev,1))` | Continuous frontier velocity |
| 21 | `peak_to_mean` | `row_max / row_mean` | LB vs non-LB decision |
| 22 | `hash_pressure` | `degree_x_m / (4 × n_threads)` | PM-BHash thrash predictor |
| 23 | `spa_init_ratio` | `log(n / mv_nnz)` | Gustavson SPA cost |
| 24 | `xfer_cost_us` | `log(x_nnz × 8 / BW_GB)` | GPU↔CPU transfer cost |

---

## DTA Improvements

| Fix | Description |
|---|---|
| A: SW-UCB | Sliding window W=12; bonus doesn't decay to 0 permanently |
| B: Counterfactual gradient | Uses global-median EMA as prior when rival has no data |
| C: Forced probe | Every 10 iters, force-try oldest-used kernel (ε-greedy) |
| D: Adaptive momentum | `β(t) = 0.9 × exp(-|grad| × 0.1)` — dampens oscillation |
| E: Per-threshold lr | t_x: 0.05/0.995 (slow), t_m: 0.03/0.99, t_lb: 0.01/0.98 (fast) |
| F: t_var threshold | 4th adaptive threshold on `degree_var` for LB choice |

---

## Build

```powershell
# From CGA-2/
mkdir build_improved
cd build_improved
cmake ../improved -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . --target cga_bfs_improved
cmake --build . --target test_models_improved
./test_models_improved
```

## Run (improved binary)

```powershell
./cga_bfs_improved data/test/graph.mtx 0 --mode dta --runs 3
./cga_bfs_improved data/test/graph.mtx 0 --mode cga
./cga_bfs_improved data/test/graph.mtx 0 --mode baseline
```

## Python training pipeline

```powershell
# 1. Generate graphs (run once)
python scripts/generate_graphs.py

# 2. Train improved models (25-dim features)
python improved/scripts/train_models.py

# 3. (Optional) Collect ground-truth timing labels from binary
python improved/scripts/collect_training_data.py

# 4. Feature extraction demo
python improved/scripts/feature_extraction.py data/train/<graph>.mtx
```
