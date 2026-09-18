# CGA-2: Sparsity-Aware Adaptive BFS Framework

A complete C++17/CUDA implementation of the research paper: **"Accelerating BFS Through a Sparsity-Aware Adaptive Framework on Heterogeneous Platforms"**.

## Features

- **Heterogeneous Execution**: Utilizes both CPUs and GPUs via the One-Copy Execution Flow.
- **5 CPU SpMSpV Kernels**: PM-BHash, LB-PM-BHash, PB-MSPA, LB-PB-MSPA, and LB-MSPA baseline (implemented via Gustavson; see KERNELS.md).
- **3 GPU SpMV/SpMSpV Kernels**: Sort-Based SpMSpV, SpMV (CSR-Vector), and Merge-Based SpMV.
- **Machine Learning Integration**: 4 decision trees (M1=3, M2=7, M3=6, M4=8) on 19 runtime features (see FEATURE_SPEC.md/SPEC.md).
- **Dynamic Threshold Adapter (DTA)**: UCB + adaptive thresholds (W=20, see SPEC.md).

## Building

Requires CMake 3.20+, C++17, CUDA Toolkit (nvcc c++17 support), and OpenMP.

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Training ML Models

Requires: `pip install -r requirements.txt` (see SPEC.md for depths/dims).

```bash
python scripts/train_models.py
```
Loads `data/train/*.mtx`, extracts 19-dim features (Table II, see
FEATURE_SPEC.md), computes heuristic-proxy timings (NOT profiled; see P0-8),
exports `.pkl` to `models_trained/` (+ `provenance.json`).

> NOTE (P0-1): C++ binaries use hardcoded heuristic rules derived from
> training; `.pkl` files are NOT loaded at runtime. See SPEC.md/KERNELS.md.

## Running

```bash
# Run with DTA (Dynamic Threshold Adapter)
./build/Release/cga_bfs data/train/graph.mtx 0 --mode dta

# Run pure paper execution flow parameters (Static ML)
./build/Release/cga_bfs data/train/graph.mtx 0 --mode cga

# Run baseline (CPU Gustavson MSPA)
./build/Release/cga_bfs data/train/graph.mtx 0 --mode baseline
```
