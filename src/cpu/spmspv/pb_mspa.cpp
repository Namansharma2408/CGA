
#include "cpu/spmspv/pb_mspa.h"
#include "cpu/spmspv/kernels.h"
#include "cpu/utils/bucket.h"
#include "cpu/utils/load_balance.h"
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <vector>

static KernelResult _pb_mspa_core(
    const CSCMatrix& A,
    std::vector<int> xi,
    std::vector<float> xv,
    const std::vector<bool>& mask,
    int n_buckets)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    int m = A.n_rows;
    int xk = (int)xi.size();

    std::vector<Bucket> buckets(n_buckets);

    #pragma omp parallel
    {
        std::vector<Bucket> local_bkts(n_buckets);

        #pragma omp for schedule(dynamic, 1) nowait
        for (int k = 0; k < xk; ++k) {
            int j    = xi[k];
            float xj = xv[k];
            int  start = A.indptr[j], end = A.indptr[j + 1];
            for (int p = start; p < end; ++p) {
                int   i   = A.indices[p];
                float aij = A.data[p];
                int b = bucket_index(i, n_buckets, m);
                local_bkts[b].emplace_back(i, aij * xj);
            }
        }

        #pragma omp critical
        {
            for (int b = 0; b < n_buckets; ++b)
                for (auto& item : local_bkts[b])
                    buckets[b].push_back(item);
        }
    }

    std::vector<float>  spa_val(m, 0.0f);
    std::vector<int8_t> spa_state(m, 0);

    for (int i = 0; i < m; ++i)
        if (mask[i]) spa_state[i] = 1;

    std::vector<int> touched;
    touched.reserve(512);

    for (int b = 0; b < n_buckets; ++b) {
        for (auto& [i, val] : buckets[b]) {
            if (spa_state[i] == 0) continue;
            if (spa_state[i] == 1) {
                spa_val[i]   = val;
                spa_state[i] = 2;
                touched.push_back(i);
            } else {
                spa_val[i] += val;
            }
        }
    }

    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());

    int nnz_out = (int)touched.size();
    std::vector<int>   out_idx(nnz_out);
    std::vector<float> out_val(nnz_out);
    for (int k = 0; k < nnz_out; ++k) {
        out_idx[k] = touched[k];
        out_val[k] = spa_val[touched[k]];
        spa_val[touched[k]]   = 0.0f;
        spa_state[touched[k]] = 0;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

    KernelResult res;
    res.y = SparseVector(m, std::move(out_idx), std::move(out_val));
    res.exec_ms = ms;
    return res;
}

KernelResult cpu_pb_mspa(
    const CSCMatrix& A, const SparseVector& x,
    const std::vector<bool>& mask, int n_buckets)
{
    int nb = n_buckets > 0 ? n_buckets : default_n_buckets(omp_get_max_threads());
    return _pb_mspa_core(A, x.indices, x.values, mask, nb);
}
