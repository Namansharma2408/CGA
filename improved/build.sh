#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# CGA-BFS  —  Improved Build
# All improvements over the research paper:
#
# Features : 25-dim  (19 paper + 6 new: F1-F6)
# Bug fixes: CSC col_counts, UCB decay, DTA one-copy, random training labels
# DTA      : SW-UCB, forced probe, counterfactual gradient,
#            adaptive momentum, per-threshold lr, t_var 4th threshold
# Model-0  : pre-filter (x_nnz=1 → PM-BHash; x_nnz>0.3n → SpMV)
# Training : real features, argmin timing labels, Eq.6 weights, depth weights
# Output   : ../build/Release/cga_bfs_improved
#
# Usage:
#   cd improved && bash build.sh
#   CUDA_ARCH=75 bash build.sh   (for RTX 20xx)
# ═══════════════════════════════════════════════════════════════════════════════

set -e
CUDA_ARCH=${CUDA_ARCH:-86}
# P2-6: resolve from script location (works from repo root or improved/).
IMPR="$(cd "$(dirname "$0")" && pwd)"   # = .../improved/
ROOT="$(dirname "${IMPR}")"             # parent = repo root

echo "═══════════════════════════════════════════════════════"
echo "  CGA-BFS  Improved Build  |  sm_${CUDA_ARCH}  |  25-dim"
echo "═══════════════════════════════════════════════════════"
mkdir -p "${ROOT}/build/Release"

# Include order matters (P2-3: same file list as improved/CMakeLists.txt):
#   -I improved/ first → picks up DIM=25 models/features.h & fixed inference.h
#   -I src/ second → all other headers from original

nvcc -O3 -std=c++17 --extended-lambda \
    -I"${IMPR}" \
    -I"${ROOT}/src" \
    -Xcompiler "-fopenmp" \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    "${IMPR}/src/main.cpp" \
    "${IMPR}/core/bfs.cpp" \
    "${ROOT}/src/core/scheduler.cpp" \
    "${ROOT}/src/core/execution_flow.cpp" \
    "${ROOT}/src/cpu/spmspv/pm_bhash.cpp" \
    "${ROOT}/src/cpu/spmspv/lb_pm_bhash.cpp" \
    "${ROOT}/src/cpu/spmspv/pb_mspa.cpp" \
    "${ROOT}/src/cpu/spmspv/lb_pb_mspa.cpp" \
    "${ROOT}/src/cpu/spmspv/gustavson.cpp" \
    "${ROOT}/src/cpu/utils/hash_table.cpp" \
    "${ROOT}/src/cpu/utils/bucket.cpp" \
    "${ROOT}/src/cpu/utils/load_balance.cpp" \
    "${ROOT}/src/matrix/csr.cpp" \
    "${ROOT}/src/matrix/csc.cpp" \
    "${ROOT}/src/matrix/loader.cpp" \
    "${IMPR}/models/features.cpp" \
    "${IMPR}/models/inference.cpp" \
    "${ROOT}/src/models/model1_platform.cpp" \
    "${ROOT}/src/models/model2_cpu.cpp" \
    "${ROOT}/src/models/model3_gpu.cpp" \
    "${ROOT}/src/models/model4_universal.cpp" \
    "${ROOT}/src/utils/timer.cpp" \
    "${ROOT}/src/utils/logger.cpp" \
    "${ROOT}/src/vector/dense_vector.cpp" \
    "${ROOT}/src/vector/sparse_vector.cpp" \
    "${ROOT}/src/vector/conversion.cpp" \
    "${ROOT}/src/gpu/spmspv/sort_based.cu" \
    "${ROOT}/src/gpu/spmv/csr_vector.cu" \
    "${ROOT}/src/gpu/spmv/merge_based.cu" \
    "${ROOT}/src/gpu/utils/memory.cu" \
    "${ROOT}/src/gpu/utils/conversion.cu" \
    -L/usr/local/cuda/lib64 -lcusparse -lcublas \
    -o "${ROOT}/build/Release/cga_bfs_improved"

echo ""
echo "✓ Binary ready: build/Release/cga_bfs_improved"
echo ""
echo "─── Improved Python Training Pipeline ──────────────────"
echo " 1. Generate graphs (run once from CGA-2/ root):"
echo "      python3 ../scripts/generate_graphs.py"
echo ""
echo " 2. Train 4 improved models (25-dim, Eq.6 weights, heuristic-proxy labels):"
echo "      python3 scripts/train_models.py   (from improved/; or python3 improved/scripts/train_models.py from root)"
echo "      # saves → improved/models_trained/*.pkl (+ provenance.json)"
echo ""
echo " 3. (Optional) Collect measured timing labels from binary:"
echo "      python3 scripts/collect_training_data.py"
echo "      # see improved/scripts/collect_training_data.py for output path"
echo ""
echo " 4. Extract 25-dim features from a graph:"
echo "      python3 scripts/feature_extraction.py <graph.mtx>"
echo "      python3 scripts/feature_extraction.py <graph.mtx> --csv"
echo ""
echo " 5. Run BFS inference (from CGA-2/ root):"
echo "      ../build/Release/cga_bfs_improved  <graph.mtx> <src> --mode baseline"
echo "      ../build/Release/cga_bfs_improved  <graph.mtx> <src> --mode cga"
echo "      ../build/Release/cga_bfs_improved  <graph.mtx> <src> --mode dta --runs 3"
echo "────────────────────────────────────────────────────────"
