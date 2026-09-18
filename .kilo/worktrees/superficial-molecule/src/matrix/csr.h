#pragma once
#include <vector>
#include <cstdint>
#include <string>

struct CSRMatrix {
    int n_rows{0};
    int n_cols{0};
    int nnz{0};

    std::vector<int>   indptr;
    std::vector<int>   indices;
    std::vector<float> data;

    CSRMatrix() = default;
    CSRMatrix(int n_rows, int n_cols, int nnz);

    float get(int row, int col) const;

    void print_info(const std::string& label = "") const;
};
