#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include "csr.h"

struct CSCMatrix {
    int n_rows{0};
    int n_cols{0};
    int nnz{0};

    std::vector<int>   indptr;
    std::vector<int>   indices;
    std::vector<float> data;

    CSCMatrix() = default;
    CSCMatrix(int n_rows, int n_cols, int nnz);

    static CSCMatrix from_csr(const CSRMatrix& A);

    void print_info(const std::string& label = "") const;
};
