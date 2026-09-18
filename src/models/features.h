#pragma once
#include <array>
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

// Canonical 19-dim Table II indices (see FEATURE_SPEC.md).
enum FeatureIdx {
    F_LOG_N = 0, F_LOG_NNZ = 1, F_LOG_AVG_DEG = 2,
    F_ROW_CV = 3, F_COL_CV = 4, F_LOG_MAX_ROW = 5,
    F_GINI_ROW = 6, F_GINI_COL = 7, F_IS_SCALE_FREE = 8,
    F_LOG_X_NNZ = 9, F_X_DENSITY = 10, F_LOG_M_NNZ = 11,
    F_M_DENSITY = 12, F_X_DECREASING = 13,
    F_LOG_NNZ_RATIO = 14, F_LOG_VALID_NNZ = 15,
    F_DEGREE_X = 16, F_DEGREE_X_M = 17, F_LOG_MASKED_VALID = 18
};

struct FeatureVector {
    static constexpr int DIM = 19;
    float v[DIM]{};

    float& operator[](int i) {
        assert(i >= 0 && i < DIM);
        if (i < 0 || i >= DIM) throw std::out_of_range("FeatureVector index");
        return v[i];
    }
    float operator[](int i) const {
        assert(i >= 0 && i < DIM);
        if (i < 0 || i >= DIM) throw std::out_of_range("FeatureVector index");
        return v[i];
    }
    // Named accessors (preferred over magic indices).
    float x_density() const { return v[F_X_DENSITY]; }
    float m_density() const { return v[F_M_DENSITY]; }
    float degree_x() const { return v[F_DEGREE_X]; }
    float row_cv() const { return v[F_ROW_CV]; }
    float col_cv() const { return v[F_COL_CV]; }
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

// NOTE: caller must pass BOTH CSR and CSC indptr arrays (P0-3 fix).
// m_density is defined as m_nnz / nnz (see FEATURE_SPEC.md, P0-2).
MatrixStats compute_matrix_stats(const int* csr_indptr, const int* csc_indptr, int n, int nnz);

float gini_coeff(const std::vector<int>& counts);

FeatureVector build_feature_vector(
    const MatrixStats& ms,
    int x_nnz, int m_nnz, int x_nnz_prev);
