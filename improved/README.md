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
| 7 | `train_models.py` | Heuristic-proxy labels (documented; P0-8) | Real stats + heuristic-proxy argmin (measured via collect_training_data.py optional) |

---

## New Features (indices 19-24; see FEATURE_SPEC.md)

| Idx | Name | Formula | Why |
|---|---|---|---|
| 19 | `degree_variance` | `row_std²` | Hub dominance beyond Gini |
| 20 | `hash_pressure` | `min(1024, x_nnz/16+1)/1024` | PM-BHash thrash predictor |
| 21 | `sparsity_aware_gini` | `gini_row*(1-xd)` | Sparsity-aware imbalance |
| 22 | `load_balance_factor` | `avg_degree*(0.5+0.5*md)` | LB vs non-LB decision |
| 23 | `growth_rate` | `log1p(x_nnz/(x_prev+1))` | Frontier velocity |
| 24 | `x_m_density` | `xd*md` | Joint sparsity |

Retired: `xfer_cost_us` (was constant 0.15, zero variance) — use `BANDWIDTH_GBS` separately if needed.

---

## DTA Improvements

| Fix | Description |
|---|---|
| A: SW-UCB | Sliding window W=20 (code UCB_WINDOW=20; see SPEC.md) |
| B: Counterfactual gradient | Uses global-median EMA as prior when rival has no data (GRADIENT_DTA) |
| C: Forced probe | Every 10 iters, oldest kernel (round-robin on ties; respects Model-0) |
| D: Adaptive momentum | `β(t) = 0.9 × exp(-|grad| × 0.1)` — dampens oscillation |
| E: Per-threshold lr | t_x: 0.05/0.995, t_m: 0.025/0.99, t_lb: 0.015/0.98 (see SPEC.md) |
| F: t_var threshold | 4th threshold on `degree_variance` at index 19 (see FEATURE_SPEC.md) |

---

## Build (bash; canonical output `build/Release/cga_bfs_improved`, see P2-2)

```bash
# From repo root — nvcc path (canonical):
bash improved/build.sh
# or via wrapper:
bash build_improved.sh

# CMake path (same file list, see improved/CMakeLists.txt):
cmake -S improved -B improved/build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build improved/build --target cga_bfs_improved
cmake --build improved/build --target test_models_improved
./improved/build/test_models_improved
```

## Run (improved binary)

```bash
./build/Release/cga_bfs_improved data/test/graph.mtx 0 --mode dta --runs 3
./build/Release/cga_bfs_improved data/test/graph.mtx 0 --mode cga
./build/Release/cga_bfs_improved data/test/graph.mtx 0 --mode baseline
./build/Release/cga_bfs_improved --help   # all flags incl. --dta-type/--graphs/--output
```

## Python training pipeline (all from repo root)

```bash
# 1. Generate graphs (run once)
python scripts/generate_graphs.py

# 2. Train improved models (25-dim features)
python improved/scripts/train_models.py
# → improved/models_trained/*.pkl (+ provenance.json)

# 3. (Optional) Collect measured timing labels from binary
python improved/scripts/collect_training_data.py

# 4. Feature extraction demo
python improved/scripts/feature_extraction.py data/train/<graph>.mtx
```
