# CGA-2: System Flow & Architecture

## Overview

The CGA-2 system is a **Sparsity-Aware Adaptive BFS Framework** on heterogeneous
CPU+GPU platforms. The chain below shows the complete pipeline from raw graph to
optimal kernel execution — for both the **basic** and **improved** versions.

---

## Architecture Chain

```mermaid
graph TD
    A[Raw Graph\n.mtx / .npz] --> B[Feature Extraction\ncompute_matrix_stats]
    B --> C[19-dim Feature Vector\nTable II paper\nM=9, V=5, H=5]
    C -->|Improved| C2[25-dim Feature Vector\n+6 new (see FEATURE_SPEC.md):\ndegree_variance,\nhash_pressure,\nsparsity_aware_gini,\nload_balance_factor,\ngrowth_rate,\nx_m_density]

    C --> D[Model 0\nPre-Filter\nFrontier=1 → PM-BHash\nDense > 30% → SpMV]
    C2 --> D

    D -->|else| E1[Model 1\nPlatform Selector\nDecision Tree depth=3\nCPU or GPU]

    E1 -->|CPU| E2[Model 2\nCPU Kernel Selector\nDecision Tree depth=7\nPM-BHash / LB-PM-BHash /\nPB-MSPA / LB-PB-MSPA /\nLB-MSPA (Gustavson impl; see KERNELS.md)]
    E1 -->|GPU| E3[Model 3\nGPU Kernel Selector\nDecision Tree depth=6\nSort-Based SpMSpV / SpMV /\nMerge-Based SpMV]
    E1 --> E4[Model 4\nUniversal One-Copy\nDecision Tree depth=8\nLB-MSPA or SpMV]

    E2 & E3 & E4 -->|concurrent| F[Inference Engine\nselect_kernel]

    F --> G[Dynamic Threshold Adapter\nDTA]

    G -->|SW-UCB| H1[Sliding Window UCB\nKernel Exploration]
    G -->|Counterfactual| H2[Gradient Update\nThreshold Adaptation]
    G -->|Forced Probe| H3[ε-probe every 10 iters\nPrevents Lock-in]
    G -->|Momentum Decay| H4[Adaptive β\nReduces Oscillation]

    H1 & H2 & H3 & H4 --> I[Selected Kernel]

    I -->|CPU| J1[PM-BHash / LB-PM-BHash\nPB-MSPA / LB-PB-MSPA\nGustavson]
    I -->|GPU| J2[SpMV / SpMSpV variants]
    I -->|One-Copy Trigger| J3[GPU → CPU Transfer\nContinue on CPU]

    J1 & J2 & J3 --> K[BFS Result\nTotal Time, Visited Nodes,\nIteration Records]
    K --> L[Feedback Loop\nUpdate DTA EMA Stats\nThreshold Gradient]
    L --> G
```

---

## Component Breakdown

### 1. Feature Extraction (`scripts/feature_extraction.py`)

| Category | Dim | Features |
|----------|-----|---------|
| M (Matrix) | 9 | log_n, log_nnz, log_avg_degree, row_cv, col_cv, log_max_row_nnz, gini_row, gini_col, is_scale_free |
| V (Vector) | 5 | log_x_nnz, x_density, log_m_nnz, m_density (=m_nnz/nnz, P0-2), x_decreasing |
| H (Hybrid) | 5 | log_nnz_ratio, log_valid_nnz, degree_x, degree_x_m, log_masked_valid_nnz |
| **NEW** 19-24 | 6 | degree_variance, hash_pressure, sparsity_aware_gini, load_balance_factor, growth_rate, x_m_density (see FEATURE_SPEC.md) |
| **Total** | **19 / 25** | |

### 2. Model Hierarchy (`src/models/`)

```
Inference Engine
├── Model 0 (pre-filter)   — frontier==1 or dense>30% → instant decision
├── Model 1 (platform)     — depth=3, CPU vs GPU
├── Model 2 (cpu kernel)   — depth=7, 5 CPU kernels [improved: cost-sensitive ensemble]
├── Model 3 (gpu kernel)   — depth=6, 3 GPU kernels
└── Model 4 (universal)    — depth=8, LB-MSPA vs SpMV (one-copy trigger)
```

### 3. DTA Algorithm (`src/models/inference.cpp`)

```
DTA State per kernel:
  ema_ms      – Exponential moving average of execution time
  n_obs       – Observation count
  last_used   – Iteration when kernel was last tried
  window[]    – Sliding window of last W observations (IMPROVED)

Per-iteration update:
  1. SW-UCB score: EMA_k - c·√(ln(min(t,W)) / window_obs(k))
  2. Select: argmin(UCB_score)   [min time = best]
  3. Counterfactual gradient → update t_x, t_m, t_lb, [t_var IMPROVED]
  4. Every 10 iters: force-probe oldest kernel (ε-greedy fallback)
  5. Momentum: β(t) = 0.9·exp(-|grad|·0.1)   [IMPROVED]

Thresholds (adaptive):
  t_x  : x_density boundary (SpMV vs SpMSpV)    lr=0.05, decay=0.995
  t_m  : m_density boundary (MSPA vs PM-BHash)  lr=0.03, decay=0.99
  t_lb : degree_x (LB variant threshold)         lr=0.01, decay=0.98
  t_var: degree_variance (NEW, improved only)    lr=0.02, decay=0.99
```

### 4. Execution Flow (`src/core/`)

```
bfs(source):
  for each depth d = 0, 1, ..., max_depth:
    1. Extract features from current frontier x and masked matrix M
    2. IF mode == STATIC:  run Models 0→1→[2 or 3]→4 concurrently
       IF mode == DTA:     use Model 1 for platform, DTA for kernel
    3. Execute selected kernel → IterationRecord{kernel, ms, overhead_ms}
    4. Update DTA stats (EMA, gradient, window)
    5. Apply one-copy trigger if x_decreasing and on_gpu
  return BFSResult{total_ms, iters[], visited[]}
```

---

## Data Flow Summary

```
Graph (.mtx)
    ↓
matrix_stats dict (O(n + nnz), once per graph)
    ↓
per-iteration feature vector (O(1), each BFS depth)
    ↓
Model chain → kernel name + platform
    ↓
kernel execution → time (ms)
    ↓
DTA update (EMA + gradient)
    ↓
BFS result / benchmark CSV
    ↓
Plot scripts → outputs/paper_plots/ | results/ | 3d_viz/
```

---

## Improvements vs Basic CGA

| Component | Basic | Improved | Impact |
|-----------|-------|----------|--------|
| Feature dim | 19 | 25 | +accuracy |
| Training labels | Heuristic proxy (NOT profiled) | Heuristic proxy (NOT profiled; collect_training_data.py optional for measured) | **Highest if measured** |
| Sample weights | Eq.6 only | Eq.6 × depth × cost-matrix | High |
| Model 0 | ✗ | ✓ pre-filter | Low-medium |
| Model depth | All=6 | M1=3, M2=7, M3=6, M4=8 | Medium |
| DTA UCB | Total-count | Sliding window (W=20) | High |
| Threshold gradient | Stale on no data | Counterfactual prior | High |
| Exploration | explore_left only | + forced probe (10-iter) | High |
| Threshold momentum | Fixed 0.9 | Dampened β(t) | Medium |
| Learning rates | Same for all | Per-threshold (0.05/0.03/0.01) | Medium |
| Inference | Sequential | Concurrent (std::async) | Medium |
| Feature caching | ✗ | Static features cached once | Low-medium |
| One-copy in DTA | ✗ | ✓ x_decreasing check | Medium |
