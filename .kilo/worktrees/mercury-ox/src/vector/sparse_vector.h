#pragma once
#include <vector>
#include <cstdint>

struct SparseVector {
    int n{0};
    int nnz{0};

    std::vector<int>   indices;
    std::vector<float> values;

    SparseVector() = default;
    explicit SparseVector(int n);
    SparseVector(int n, std::vector<int> idx, std::vector<float> vals);

    void push(int idx, float val);
    void sort_indices();
    void clear();
    bool empty() const { return nnz == 0; }
    float get_density() const;
};
