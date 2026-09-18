#include "features.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>
#include <tuple>

float gini_coeff(const std::vector<int>& counts) {
    if (counts.empty()) return 0.0f;
    std::vector<float> v(counts.begin(), counts.end());
    std::sort(v.begin(), v.end());
    float n    = (float)v.size();
    float sum  = std::accumulate(v.begin(), v.end(), 0.0f);
    if (sum == 0.0f) return 0.0f;
    float num  = 0.0f;
    for (int i = 0; i < (int)v.size(); ++i)
        num += (2.0f * (i + 1) - n - 1) * v[i];
    return num / (n * sum);
}

MatrixStats compute_matrix_stats(const int* indptr, int n, int nnz) {
    MatrixStats ms;
    ms.n = n; ms.nnz = nnz;
    ms.avg_degree = n > 0 ? (float)nnz / n : 0.0f;

    std::vector<int> row_counts(n);
    for (int i = 0; i < n; ++i)
        row_counts[i] = indptr[i+1] - indptr[i];

    float sum = 0, sum2 = 0, mx = 0;
    for (int x : row_counts) { 
        sum += x; 
        sum2 += (float)x*x; 
        if (x > mx) mx = x; 
    }
    ms.row_mean = n > 0 ? sum / n : 0.0f;
    ms.row_max = mx;
    ms.row_std = n > 1 ? std::sqrt(std::max(0.0f, (sum2 - (sum*sum/n)) / (n-1))) : 0.0f;
    ms.degree_variance = ms.row_std * ms.row_std;

    ms.col_mean = ms.row_mean; ms.col_std = ms.row_std; ms.col_max = mx;
    ms.gini_row = gini_coeff(row_counts);
    ms.gini_col = ms.gini_row;
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

    f[0] = std::log1p(n);
    f[1] = std::log1p((float)ms.nnz);
    f[2] = std::log1p(ad);
    f[3] = row_cv;
    f[4] = ms.degree_variance; // New: degree_variance for Fix F
    f[5] = std::log1p(ms.row_max);
    f[6] = ms.gini_row;
    f[7] = ms.gini_col;
    f[8] = ms.is_scale_free ? 1.0f : 0.0f;
    f[9]  = std::log1p((float)x_nnz);
    f[10] = xd;
    f[11] = std::log1p((float)m_nnz);
    f[12] = md;
    f[13] = (x_nnz < x_nnz_prev && x_nnz_prev > 0) ? 1.0f : 0.0f;
    
    float nnz_ratio   = m_nnz > 0 ? (float)x_nnz / m_nnz : 0.0f;
    float valid_nnz   = (float)ms.nnz * xd;
    float degree_x    = ad * xd;
    float degree_x_m  = ad * xd * md;
    float mv_nnz      = (float)ms.nnz * xd * md;
    
    f[14] = std::log1p(nnz_ratio  + 1e-9f);
    f[15] = std::log1p(valid_nnz  + 1.0f);
    f[16] = degree_x;
    f[17] = degree_x_m;
    f[18] = std::log1p(mv_nnz + 1.0f);

    int active_buckets = std::min(1024, (int)(x_nnz / 16 + 1));
    f[19] = (float)active_buckets / 1024.0f;
    f[20] = ms.gini_row * (1.0f - xd);
    f[21] = ad * (0.5f + 0.5f * md);
    float growth = (x_nnz_prev > 0) ? (float)x_nnz / (x_nnz_prev + 1) : 1.0f;
    f[22] = std::log1p(growth);
    f[23] = xd * md;
    f[24] = 0.15f; // Hardware transfer cost US baseline

    return f;
}
