#ifndef CPU_SPMSPV_KERNELS_H
#define CPU_SPMSPV_KERNELS_H

#include "matrix/csc.h"
#include "vector/sparse_vector.h"
#include <vector>

struct KernelResult {
    SparseVector y;
    double exec_ms{0.0};
};

KernelResult cpu_pm_bhash(
    const CSCMatrix& A,
    const SparseVector& x,
    const std::vector<bool>& mask,
    int n_buckets = -1);

KernelResult cpu_lb_pm_bhash(
    const CSCMatrix& A,
    const SparseVector& x,
    const std::vector<bool>& mask,
    int n_buckets = -1);

KernelResult cpu_pb_mspa(
    const CSCMatrix& A,
    const SparseVector& x,
    const std::vector<bool>& mask,
    int n_buckets = -1);

KernelResult cpu_lb_pb_mspa(
    const CSCMatrix& A,
    const SparseVector& x,
    const std::vector<bool>& mask,
    int n_buckets = -1);

KernelResult cpu_gustavson(
    const CSCMatrix& A,
    const SparseVector& x,
    const std::vector<bool>& mask);

#endif
