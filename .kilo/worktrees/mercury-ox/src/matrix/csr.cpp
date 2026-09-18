#include "matrix/csr.h"
#include <algorithm>
#include <iostream>

CSRMatrix::CSRMatrix(int n_rows, int n_cols, int nnz)
    : n_rows(n_rows), n_cols(n_cols), nnz(nnz)
{
    indptr.resize(n_rows + 1, 0);
    indices.resize(nnz, 0);
    data.resize(nnz, 0.0f);
}

float CSRMatrix::get(int row, int col) const {
    for (int k = indptr[row]; k < indptr[row + 1]; ++k)
        if (indices[k] == col) return data[k];
    return 0.0f;
}

void CSRMatrix::print_info(const std::string& label) const {
    std::cout << "[CSR" << (label.empty() ? "" : " " + label) << "] "
              << "n=" << n_rows << " nnz=" << nnz
              << " avg_deg=" << (n_rows > 0 ? (double)nnz / n_rows : 0.0) << "\n";
}
