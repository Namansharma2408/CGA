
#include "cpu/spmspv/gustavson.h"
#include "cpu/spmspv/kernels.h"
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <vector>

KernelResult cpu_gustavson(
    const CSCMatrix& A, const SparseVector& x,
    const std::vector<bool>& mask)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    int m  = A.n_rows;
    int xk = x.nnz;

    std::vector<float>  spa_val(m, 0.0f);
    std::vector<bool>   spa_touched(m, false);
    std::vector<int>    touched;
    touched.reserve(1024);

    for (int k = 0; k < xk; ++k) {
        int   j  = x.indices[k];
        float xj = x.values[k];
        int   start = A.indptr[j], end = A.indptr[j + 1];
        for (int p = start; p < end; ++p) {
            int   i   = A.indices[p];
            float aij = A.data[p];
            if (mask[i]) {
                spa_val[i] += aij * xj;
                if (!spa_touched[i]) {
                    spa_touched[i] = true;
                    touched.push_back(i);
                }
            }
        }
    }

    std::sort(touched.begin(), touched.end());
    int nnz_out = (int)touched.size();
    std::vector<int>   out_idx(nnz_out);
    std::vector<float> out_val(nnz_out);
    for (int k = 0; k < nnz_out; ++k) {
        int i = touched[k];
        out_idx[k]     = i;
        out_val[k]     = spa_val[i];
        spa_val[i]     = 0.0f;
        spa_touched[i] = false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

    KernelResult res;
    res.y = SparseVector(m, std::move(out_idx), std::move(out_val));
    res.exec_ms = ms;
    return res;
}
