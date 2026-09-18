#include "vector/sparse_vector.h"
#include <algorithm>
#include <numeric>
#include <cassert>

SparseVector::SparseVector(int n) : n(n) {}

SparseVector::SparseVector(int n, std::vector<int> idx, std::vector<float> vals)
    : n(n), nnz((int)idx.size()), indices(std::move(idx)), values(std::move(vals)) {}

void SparseVector::push(int idx, float val) {
    indices.push_back(idx);
    values.push_back(val);
    ++nnz;
}

void SparseVector::sort_indices() {
    std::vector<int> perm(nnz);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(),
              [&](int a, int b){ return indices[a] < indices[b]; });
    std::vector<int>   si(nnz);
    std::vector<float> sv(nnz);
    for (int i = 0; i < nnz; ++i) {
        si[i] = indices[perm[i]];
        sv[i] = values[perm[i]];
    }
    indices = std::move(si);
    values  = std::move(sv);
}

void SparseVector::clear() {
    indices.clear(); values.clear(); nnz = 0;
}

float SparseVector::get_density() const {
    return n > 0 ? (float)nnz / n : 0.0f;
}
