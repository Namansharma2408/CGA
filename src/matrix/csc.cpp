#include "matrix/csc.h"
#include <iostream>
#include <algorithm>

CSCMatrix::CSCMatrix(int n_rows, int n_cols, int nnz)
    : n_rows(n_rows), n_cols(n_cols), nnz(nnz)
{
    indptr.resize(n_cols + 1, 0);
    indices.resize(nnz, 0);
    data.resize(nnz, 0.0f);
}

CSCMatrix CSCMatrix::from_csr(const CSRMatrix& A) {
    int n = A.n_rows, m = A.n_cols, nnz = A.nnz;
    CSCMatrix B(n, m, nnz);

    for (int k = 0; k < nnz; ++k)
        B.indptr[A.indices[k] + 1]++;
    for (int j = 0; j < m; ++j)
        B.indptr[j + 1] += B.indptr[j];

    std::vector<int> pos(B.indptr.begin(), B.indptr.begin() + m);
    for (int i = 0; i < n; ++i) {
        for (int k = A.indptr[i]; k < A.indptr[i + 1]; ++k) {
            int j = A.indices[k];
            int p = pos[j]++;
            B.indices[p] = i;
            B.data[p]    = A.data[k];
        }
    }
    return B;
}

void CSCMatrix::print_info(const std::string& label) const {
    std::cout << "[CSC" << (label.empty() ? "" : " " + label) << "] "
              << "n=" << n_cols << " nnz=" << nnz
              << " avg_deg=" << (n_cols > 0 ? (double)nnz / n_cols : 0.0) << "\n";
}
