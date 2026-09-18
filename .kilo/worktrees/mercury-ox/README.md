# CGA-2: Sparsity-Aware Adaptive BFS Framework

A complete C++17/CUDA implementation of the research paper: **"Accelerating BFS Through a Sparsity-Aware Adaptive Framework on Heterogeneous Platforms"**.

## Features

- **Heterogeneous Execution**: Utilizes both CPUs and GPUs via the One-Copy Execution Flow.
- **5 CPU SpMSpV Kernels**: PM-BHash, LB-PM-BHash, PB-MSPA, LB-PB-MSPA, and Gustavson's Algorithm.
- **3 GPU SpMV/SpMSpV Kernels**: Sort-based SpMSpV (Thrust radix sort), Masked CSR Vector SpMV (warp-per-row), and Masked Merge-Based SpMV (load balanced).
- **Machine Learning Integration**: 4 Decision trees extracting 19 runtime graph features to predict the optimal platform and kernel for each BFS iteration.
- **Dynamic Threshold Adapter (DTA)**: Custom UCB + adaptive gradient-free thresholding algorithm to detect hardware limits without offline profiling.

## Building

Requires CMake 3.20+, C++17, CUDA Toolkit (nvcc c++17 support), and OpenMP.

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Running

```bash
# Run with DTA (Dynamic Threshold Adapter)
.\cga_bfs.exe data\raw\graph.mtx 0 --mode dta

# Run pure paper execution flow parameters (Static ML)
.\cga_bfs.exe data\raw\graph.mtx 0 --mode cga

# Run baseline (CPU Gustavson MSPA)
.\cga_bfs.exe data\raw\graph.mtx 0 --mode baseline
```

## Training ML Models

Ensure you're using a python environment with `scikit-learn` and `numpy`:
```bash
python scripts/train_models.py
```
This script generates synthetic graph iterations and exports trained Sklearn Decision trees into `models_trained/`.
