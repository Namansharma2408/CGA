#pragma once
#include "matrix/csr.h"
#include "vector/sparse_vector.h"
#include <vector>

struct GPUKernelResult {
    SparseVector y;
    double exec_ms{0.0};
};

GPUKernelResult gpu_sort_based_spmspv(
    const CSRMatrix& A,
    const SparseVector& x,
    const std::vector<bool>& mask);

GPUKernelResult gpu_csr_vector_spmv(
    const CSRMatrix& A,
    const SparseVector& x,
    const std::vector<bool>& mask);

GPUKernelResult gpu_merge_spmv(
    const CSRMatrix& A,
    const SparseVector& x,
    const std::vector<bool>& mask);
