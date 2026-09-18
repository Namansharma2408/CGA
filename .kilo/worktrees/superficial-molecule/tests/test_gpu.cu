#include "gpu/gpu_kernels.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "Running GPU Kernel Integration Tests...\n";

    CSRMatrix A;
    A.n_rows = 3; A.n_cols = 3; A.nnz = 3;
    A.indptr = {0, 1, 2, 3};
    A.indices = {0, 1, 2};
    A.data = {10.0f, 20.0f, 30.0f};

    SparseVector x(3, {0, 2}, {1.0f, 1.0f});
    std::vector<bool> mask = {false, true, true};

    auto verify = [&](GPUKernelResult res, const std::string& name) {
        bool pass = res.y.nnz == 1 && res.y.indices[0] == 2 && res.y.values[0] == 30.0f;
        std::cout << " - " << name << ": " << (pass ? "PASSED" : "FAILED")
                  << " (took " << res.exec_ms << "ms)\n";
        assert(pass);
    };

    verify(gpu_sort_based_spmspv(A, x, mask), "G1: Sort-based");
    verify(gpu_csr_vector_spmv(A, x, mask), "G2: CSR-vector");
    verify(gpu_merge_based_spmv(A, x, mask), "G3: Merge-based");

    std::cout << "All GPU Tests passed successfully.\n";
    return 0;
}
