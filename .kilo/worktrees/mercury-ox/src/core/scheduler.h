#pragma once
#include "matrix/csc.h"
#include "matrix/csr.h"
#include "vector/sparse_vector.h"
#include <string>
#include <vector>

SparseVector dispatch_kernel(
    const std::string& kernel_name,
    const CSRMatrix& A_csr,
    const CSCMatrix& A_csc,
    const SparseVector& x,
    const std::vector<bool>& mask,
    double& out_ms);
