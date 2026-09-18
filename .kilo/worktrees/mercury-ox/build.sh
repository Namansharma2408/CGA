#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# CGA-BFS  —  Original Research Paper Build
# Paper: "Accelerating BFS Through a Sparsity-Aware Adaptive Framework
#         on Heterogeneous Platforms"
#
# Features : 19-dim  (Table II: M×9, V×5, H×5)
# Models   : 4 decision trees (max_depth=5 each)
# DTA      : original UCB + adaptive thresholds
# Output   : build/Release/cga_bfs
#
# Usage:
#   bash build.sh
#   CUDA_ARCH=75 bash build.sh   (for RTX 20xx)
# ═══════════════════════════════════════════════════════════════════════════════

set -e
CUDA_ARCH=${CUDA_ARCH:-86}    # 86=RTX 30xx  89=RTX 40xx  75=RTX 20xx  80=A100

echo "═══════════════════════════════════════════════════════"
echo "  CGA-BFS  Original Build  |  sm_${CUDA_ARCH}  |  19-dim"
echo "═══════════════════════════════════════════════════════"
mkdir -p build/Release

nvcc -O3 -std=c++17 \
    -I"$(pwd)/src" \
    -Xcompiler "-fopenmp" \
    --extended-lambda \
    -Wno-deprecated-gpu-targets \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_86,code=compute_86 \
    src/main.cpp \
    src/core/bfs.cpp \
    src/core/scheduler.cpp \
    src/core/execution_flow.cpp \
    src/cpu/spmspv/pm_bhash.cpp \
    src/cpu/spmspv/lb_pm_bhash.cpp \
    src/cpu/spmspv/pb_mspa.cpp \
    src/cpu/spmspv/lb_pb_mspa.cpp \
    src/cpu/spmspv/gustavson.cpp \
    src/cpu/utils/hash_table.cpp \
    src/cpu/utils/bucket.cpp \
    src/cpu/utils/load_balance.cpp \
    src/matrix/csr.cpp \
    src/matrix/csc.cpp \
    src/matrix/loader.cpp \
    src/models/features.cpp \
    src/models/inference.cpp \
    src/models/model1_platform.cpp \
    src/models/model2_cpu.cpp \
    src/models/model3_gpu.cpp \
    src/models/model4_universal.cpp \
    src/utils/timer.cpp \
    src/utils/logger.cpp \
    src/vector/dense_vector.cpp \
    src/vector/sparse_vector.cpp \
    src/vector/conversion.cpp \
    src/gpu/spmspv/sort_based.cu \
    src/gpu/spmv/csr_vector.cu \
    src/gpu/spmv/merge_based.cu \
    src/gpu/utils/memory.cu \
    src/gpu/utils/conversion.cu \
    -o build/Release/cga_bfs

echo "Building isolated CPU micro-benchmark (Fig 13 evaluation)..."
g++ -O3 -std=c++17 -I"$(pwd)/src" -fopenmp \
    src/benchmark_cpu.cpp \
    src/matrix/csr.cpp src/matrix/csc.cpp src/matrix/loader.cpp \
    src/cpu/spmspv/pm_bhash.cpp src/cpu/spmspv/lb_pm_bhash.cpp \
    src/cpu/spmspv/pb_mspa.cpp src/cpu/spmspv/lb_pb_mspa.cpp src/cpu/spmspv/gustavson.cpp \
    src/cpu/utils/hash_table.cpp src/cpu/utils/bucket.cpp src/cpu/utils/load_balance.cpp \
    src/utils/timer.cpp \
    src/vector/sparse_vector.cpp src/vector/dense_vector.cpp \
    -o build/Release/benchmark_cpu

echo ""
echo "✓ Binary ready: build/Release/cga_bfs"
echo ""
echo "─── Python Training Pipeline ───────────────────────────"
echo " 1. Generate graphs:"
echo "      python3 scripts/generate_graphs.py"
echo ""
echo " 2. Train 4 decision-tree models (19-dim features):"
echo "      python3 scripts/train_models.py"
echo ""
echo " 3. Run BFS inference (three modes):"
echo "      ./build/Release/cga_bfs  <graph.mtx> <src> --mode baseline"
echo "      ./build/Release/cga_bfs  <graph.mtx> <src> --mode cga"
echo "      ./build/Release/cga_bfs  <graph.mtx> <src> --mode dta --runs 3"
echo ""
echo " 4. Full automated test pipeline:"
echo "      python3 scripts/run_inference.py"
echo ""
echo " 5. Plot results:"
echo "      python3 scripts/plot_results.py"
echo "────────────────────────────────────────────────────────"
