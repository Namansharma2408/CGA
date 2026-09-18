// P0-2/P0-3: feature parity — md=m_nnz/nnz, col_* from CSC, directed graphs.
#include "models/features.h"
#include <cmath>
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Running feature parity tests...\n";
    // 4-node directed graph: row dist != col dist.
    // CSR indptr: row0:2, row1:1, row2:0, row3:1  (nnz=4)
    // CSC indptr: col0:1, col1:2, col2:0, col3:1
    int csr_ptr[5] = {0, 2, 3, 3, 4};
    int csc_ptr[5] = {0, 1, 3, 3, 4};
    MatrixStats ms = compute_matrix_stats(csr_ptr, csc_ptr, 4, 4);
    std::cout << " row_mean=" << ms.row_mean << " col_mean=" << ms.col_mean << "\n";
    std::cout << " gini_row=" << ms.gini_row << " gini_col=" << ms.gini_col << "\n";
    // Directed: col distribution differs from row -> gini/col_cv must differ
    // from the buggy copy (row==col). Here row_counts=[2,1,0,1], col=[1,2,0,1].
    // Both have same multiset so gini equal, but test with asymmetric below.
    assert(ms.n == 4 && ms.nnz == 4);

    // Asymmetric: row=[3,0,0,0], col=[1,1,1,0] -> must differ.
    int csr2[5] = {0, 3, 3, 3, 3};
    int csc2[5] = {0, 1, 2, 3, 3};
    MatrixStats ms2 = compute_matrix_stats(csr2, csc2, 4, 3);
    std::cout << " asym gini_row=" << ms2.gini_row << " gini_col=" << ms2.gini_col << "\n";
    assert(std::fabs(ms2.gini_row - ms2.gini_col) > 1e-3 &&
           "col_counts must come from CSC (P0-3)");

    // P0-2: md = m_nnz/nnz.
    FeatureVector f = build_feature_vector(ms, 2, 2, 1);
    float expected_md = 2.0f / 4.0f;  // 0.5
    std::cout << " md=" << f[F_M_DENSITY] << " expected=" << expected_md << "\n";
    assert(std::fabs(f[F_M_DENSITY] - expected_md) < 1e-5);
    // xd = 2/4 = 0.5
    assert(std::fabs(f[F_X_DENSITY] - 0.5f) < 1e-5);
    // col_cv at index 4 (Table II).
    assert(f[F_COL_CV] >= 0.0f);

    // Bounds check throws (P3-8).
    bool threw = false;
    try { f[99] = 1.0f; } catch (const std::out_of_range&) { threw = true; }
    assert(threw && "FeatureVector OOB must throw");

    std::cout << "Feature parity tests passed.\n";
    return 0;
}
