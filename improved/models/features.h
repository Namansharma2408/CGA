#pragma once
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>


// Canonical 25-dim spec (see FEATURE_SPEC.md):
// 0-18 identical to basic Table II (F_COL_CV at 4), 19-24 new.
enum FeatureIdx25 {
    F25_LOG_N = 0, F25_LOG_NNZ = 1, F25_LOG_AVG_DEG = 2,
    F25_ROW_CV = 3, F25_COL_CV = 4, F25_LOG_MAX_ROW = 5,
    F25_GINI_ROW = 6, F25_GINI_COL = 7, F25_IS_SCALE_FREE = 8,
    F25_LOG_X_NNZ = 9, F25_X_DENSITY = 10, F25_LOG_M_NNZ = 11,
    F25_M_DENSITY = 12, F25_X_DECREASING = 13,
    F25_LOG_NNZ_RATIO = 14, F25_LOG_VALID_NNZ = 15,
    F25_DEGREE_X = 16, F25_DEGREE_X_M = 17, F25_LOG_MASKED_VALID = 18,
    F25_DEGREE_VAR = 19, F25_HASH_PRESSURE = 20, F25_SPARSE_GINI = 21,
    F25_LB_FACTOR = 22, F25_GROWTH_RATE = 23, F25_X_M_DENSITY = 24
};

// Compat: first 19 indices identical to basic Table II, so parent
// src/models/model*.cpp (using F_* names) compile unchanged in improved builds.
static constexpr int F_LOG_N = F25_LOG_N;
static constexpr int F_LOG_NNZ = F25_LOG_NNZ;
static constexpr int F_LOG_AVG_DEG = F25_LOG_AVG_DEG;
static constexpr int F_ROW_CV = F25_ROW_CV;
static constexpr int F_COL_CV = F25_COL_CV;
static constexpr int F_LOG_MAX_ROW = F25_LOG_MAX_ROW;
static constexpr int F_GINI_ROW = F25_GINI_ROW;
static constexpr int F_GINI_COL = F25_GINI_COL;
static constexpr int F_IS_SCALE_FREE = F25_IS_SCALE_FREE;
static constexpr int F_LOG_X_NNZ = F25_LOG_X_NNZ;
static constexpr int F_X_DENSITY = F25_X_DENSITY;
static constexpr int F_LOG_M_NNZ = F25_LOG_M_NNZ;
static constexpr int F_M_DENSITY = F25_M_DENSITY;
static constexpr int F_X_DECREASING = F25_X_DECREASING;
static constexpr int F_LOG_NNZ_RATIO = F25_LOG_NNZ_RATIO;
static constexpr int F_LOG_VALID_NNZ = F25_LOG_VALID_NNZ;
static constexpr int F_DEGREE_X = F25_DEGREE_X;
static constexpr int F_DEGREE_X_M = F25_DEGREE_X_M;
static constexpr int F_LOG_MASKED_VALID = F25_LOG_MASKED_VALID;

struct FeatureVector {
    static constexpr int DIM = 25;
    float v[DIM]{};

    float& operator[](int i) {
        assert(i >= 0 && i < DIM);
        if (i < 0 || i >= DIM) throw std::out_of_range("FeatureVector25 index");
        return v[i];
    }
    float operator[](int i) const {
        assert(i >= 0 && i < DIM);
        if (i < 0 || i >= DIM) throw std::out_of_range("FeatureVector25 index");
        return v[i];
    }
};

static const char* FEATURE_NAMES[25] = {
    "log_n", "log_nnz", "log_avg_degree",
    "row_cv", "col_cv", "log_max_row_nnz",
    "gini_row", "gini_col", "is_scale_free",
    "log_x_nnz", "x_density", "log_m_nnz", "m_density", "x_decreasing",
    "log_nnz_ratio", "log_valid_nnz", "degree_x", "degree_x_m", "log_masked_valid_nnz",
    "degree_variance", "hash_pressure", "sparsity_aware_gini", "load_balance_factor", "growth_rate", "x_m_density"
};

// NOTE: xfer_cost_us (was constant 0.15, zero variance) removed from the
// trained vector; use BANDWIDTH_GBS separately for cost modelling if needed.
static constexpr float BANDWIDTH_GBS = 12.0f;

struct MatrixStats {
    int   n{0}, nnz{0};
    float avg_degree{0};
    float row_mean{0}, row_std{0}, row_max{0};
    float col_mean{0}, col_std{0}, col_max{0};
    float gini_row{0}, gini_col{0};
    float degree_variance{0}; // Fix F
    bool  is_scale_free{false};
};

MatrixStats compute_matrix_stats(
    const int* csr_indptr,
    const int* csc_indptr,
    int n, int nnz);

float gini_coeff(const std::vector<int>& counts);

FeatureVector build_feature_vector(
    const MatrixStats& ms,
    int x_nnz,
    int m_nnz,
    int x_nnz_prev
);
