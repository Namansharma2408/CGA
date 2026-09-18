
#include "cpu/spmspv/lb_pm_bhash.h"
#include "cpu/spmspv/kernels.h"
#include "cpu/utils/bucket.h"
#include "cpu/utils/hash_table.h"
#include "cpu/utils/load_balance.h"
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <numeric>

KernelResult cpu_lb_pm_bhash(
    const CSCMatrix& A, const SparseVector& x,
    const std::vector<bool>& mask, int n_buckets)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    int m = A.n_rows;
    int nb = n_buckets > 0 ? n_buckets : default_n_buckets(omp_get_max_threads());

    std::vector<int>   xi(x.indices);
    std::vector<float> xv(x.values);

    static_load_balance(A, xi, xv);

    int xk = (int)xi.size();

    std::vector<Bucket> buckets(nb);
    #pragma omp parallel
    {
        std::vector<Bucket> local_bkts(nb);
        #pragma omp for schedule(dynamic, 1) nowait
        for (int k = 0; k < xk; ++k) {
            int j     = xi[k];
            float xj  = xv[k];
            int start = A.indptr[j];
            int end   = A.indptr[j + 1];
            for (int p = start; p < end; ++p) {
                int   i   = A.indices[p];
                float aij = A.data[p];
                if (mask[i]) {
                    int b = bucket_index(i, nb, m);
                    local_bkts[b].emplace_back(i, aij * xj);
                }
            }
        }
        #pragma omp critical
        {
            for (int b = 0; b < nb; ++b)
                for (auto& item : local_bkts[b])
                    buckets[b].push_back(item);
        }
    }

    std::vector<std::vector<int>>   bucket_idx(nb);
    std::vector<std::vector<float>> bucket_val(nb);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < nb; ++b) {
        if (buckets[b].empty()) continue;
        BucketHashTable ht((int)buckets[b].size());
        for (auto& [i, v] : buckets[b]) ht.insert(i, v);
        ht.collect(bucket_idx[b], bucket_val[b]);
    }

    std::vector<int> out_idx;
    std::vector<float> out_val;
    for (int b = 0; b < nb; ++b) {
        for (size_t i = 0; i < bucket_idx[b].size(); ++i) {
            out_idx.push_back(bucket_idx[b][i]);
            out_val.push_back(bucket_val[b][i]);
        }
    }

    std::vector<int> perm(out_idx.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int a, int b){ return out_idx[a] < out_idx[b]; });
    std::vector<int>   si(out_idx.size());
    std::vector<float> sv(out_val.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        si[i] = out_idx[perm[i]];
        sv[i] = out_val[perm[i]];
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

    KernelResult res;
    res.y = SparseVector(m, std::move(si), std::move(sv));
    res.exec_ms = ms;
    return res;
}
