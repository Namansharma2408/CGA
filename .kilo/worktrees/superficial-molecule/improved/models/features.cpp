#include "features.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>
#include <tuple>
#include <omp.h>

float gini_coeff(const std::vector<int>& counts) {
    if (counts.empty()) return 0.0f;
    std::vector<float> v(counts.begin(), counts.end());
    std::sort(v.begin(), v.end());
    float n   = (float)v.size();
    float sum = std::accumulate(v.begin(), v.end(), 0.0f);
    if (sum == 0.0f) return 0.0f;
    float num = 0.0f;
    for (int i = 0; i < (int)v.size(); ++i)
        num += (2.0f * (i + 1) - n - 1) * v[i];
    return num / (n * sum);
}

MatrixStats compute_matrix_stats(
    const int* csr_indptr,
    const int* csc_indptr,
    int n, int nnz)
{
    MatrixStats ms;
    ms.n   = n;
    ms.nnz = nnz;
    ms.avg_degree = n > 0 ? (float)nnz / n : 0.0f;

    std::vector<int> row_counts(n), col_counts(n);
    for (int i = 0; i < n; ++i) {
        row_counts[i] = csr_indptr[i + 1] - csr_indptr[i];
        col_counts[i] = csc_indptr[i + 1] - csc_indptr[i];
    }

    auto stats = [](const std::vector<int>& c)
        -> std::tuple<float, float, float>
    {
        if (c.empty()) return {0, 0, 0};
        float sum = 0, sum2 = 0, mx = 0;
        for (int x : c) {
            sum  += x;
            sum2 += (float)x * x;
            if (x > mx) mx = x;
        }
        float mean  = sum / c.size();
        float stdv  = std::sqrt(std::max(0.0f, sum2 / c.size() - mean * mean));
        return {mean, stdv, mx};
    };

    auto [rm, rs, rmax] = stats(row_counts);
    auto [cm, cs, cmax] = stats(col_counts);
    ms.row_mean = rm; ms.row_std = rs; ms.row_max = rmax;
    ms.col_mean = cm; ms.col_std = cs; ms.col_max = cmax;
    ms.gini_row = gini_coeff(row_counts);
    ms.gini_col = gini_coeff(col_counts);
    ms.degree_variance = rs * rs; // Fix F
    ms.is_scale_free = (ms.gini_row > 0.4f || ms.avg_degree > 10.0f);
    return ms;
}

FeatureVector build_feature_vector(
    const MatrixStats& ms, int x_nnz, int m_nnz, int x_nnz_prev)
{
    FeatureVector f;
    float n  = (float)ms.n;
    float ad = ms.avg_degree;
    float xd = n > 0 ? (float)x_nnz / n : 0.0f;
    float md = n > 0 ? (float)m_nnz / n : 0.0f;
    float row_cv = ms.row_mean > 0 ? ms.row_std / ms.row_mean : 0.0f;

    f[0]  = std::log1p(n);
    f[1]  = std::log1p((float)ms.nnz);
    f[2]  = std::log1p(ad);
    f[3]  = row_cv;
    // [HIGHLIGHT] Fix F: Improved feature index 4 (degree_variance) for DTA thresholds
    f[4]  = ms.degree_variance; 
    f[5]  = std::log1p(ms.row_max);
    f[6]  = ms.gini_row;
    f[7]  = ms.gini_col;
    f[8]  = ms.is_scale_free ? 1.0f : 0.0f;

    f[9]  = std::log1p((float)x_nnz);
    f[10] = xd;
    f[11] = std::log1p((float)m_nnz);
    f[12] = md;
    f[13] = (x_nnz < x_nnz_prev && x_nnz_prev > 0) ? 1.0f : 0.0f;

    float nnz_ratio        = m_nnz > 0 ? (float)x_nnz / m_nnz : 0.0f;
    float valid_nnz        = (float)ms.nnz * xd;
    float degree_x         = ad * xd;
    float degree_x_m       = ad * xd * md;
    float mv_nnz           = (float)ms.nnz * xd * md;
    
    f[14] = std::log1p(nnz_ratio  + 1e-9f);
    f[15] = std::log1p(valid_nnz  + 1.0f);
    f[16] = degree_x;
    f[17] = degree_x_m;
    f[18] = std::log1p(mv_nnz + 1.0f);

    // [HIGHLIGHT] New Features (F1-F6) for Improved 25-dim Vector
    int active_buckets = std::min(1024, (int)(x_nnz / 16 + 1));
    f[19] = (float)active_buckets / 1024.0f; // hash_pressure
    f[20] = ms.gini_row * (1.0f - xd); // sparsity_aware_gini
    f[21] = ad * (0.5f + 0.5f * md);    // load_balance_factor
    float growth = (x_nnz_prev > 0) ? (float)x_nnz / (x_nnz_prev + 1) : 1.0f;
    f[22] = std::log1p(growth); // growth_rate
    f[23] = xd * md; // x_m_density
    f[24] = 0.15f; // xfer_cost_us (hardware constant baseline)

    return f;
}
