#pragma once
#include <array>
#include <string>
#include <vector>

struct FeatureVector {
    static constexpr int DIM = 19;
    float v[DIM]{};

    float& operator[](int i)       { return v[i]; }
    float  operator[](int i) const { return v[i]; }
};

static const char* FEATURE_NAMES[19] = {
    "log_n", "log_nnz", "log_avg_degree",
    "row_cv", "col_cv", "log_max_row_nnz",
    "gini_row", "gini_col", "is_scale_free",
    "log_x_nnz", "x_density", "log_m_nnz", "m_density", "x_decreasing",
    "log_nnz_ratio", "log_valid_nnz", "degree_x", "degree_x_m", "log_masked_valid_nnz"
};

struct MatrixStats {
    int   n{0}, nnz{0};
    float avg_degree{0};
    float row_mean{0}, row_std{0}, row_max{0};
    float col_mean{0}, col_std{0}, col_max{0};
    float gini_row{0}, gini_col{0};
    bool  is_scale_free{false};
};

MatrixStats compute_matrix_stats(const int* indptr, int n, int nnz);

float gini_coeff(const std::vector<int>& counts);

FeatureVector build_feature_vector(
    const MatrixStats& ms,
    int x_nnz, int m_nnz, int x_nnz_prev);
