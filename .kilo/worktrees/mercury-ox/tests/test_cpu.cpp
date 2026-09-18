#include "cpu/spmspv/kernels.h"
#include "matrix/csc.h"
#include "vector/sparse_vector.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "Running CPU Kernel Integration Tests...\n";

    CSCMatrix A;
    A.n_rows = 3; A.n_cols = 3; A.nnz = 3;
    A.indptr = {0, 1, 2, 3};
    A.indices = {0, 1, 2};
    A.data = {1.0f, 2.0f, 3.0f};

    SparseVector x(3, {0, 2}, {1.0f, 1.0f});

    std::vector<bool> mask = {true, true, false};

    auto verify = [&](KernelResult res, const std::string& name) {
        bool pass = res.y.nnz == 1 && res.y.indices[0] == 0 && res.y.values[0] == 1.0f;
        std::cout << " - " << name << ": " << (pass ? "PASSED" : "FAILED")
                  << " (took " << res.exec_ms << "ms)\n";
        assert(pass);
    };

    verify(cpu_pm_bhash(A, x, mask, 8), "PM-BHash");
    verify(cpu_lb_pm_bhash(A, x, mask, 8), "LB-PM-BHash");
    verify(cpu_pb_mspa(A, x, mask, 8), "PB-MSPA");
    verify(cpu_lb_pb_mspa(A, x, mask, 8), "LB-PB-MSPA");
    verify(cpu_gustavson(A, x, mask), "Gustavson LB-MSPA");

    std::cout << "All CPU Tests passed successfully.\n";
    return 0;
}
