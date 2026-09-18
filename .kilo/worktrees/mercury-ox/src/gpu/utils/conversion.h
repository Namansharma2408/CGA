#pragma once
#include <vector>

int  gpu_dense_to_sparse(const float* d_dense, int n,
                         std::vector<int>& out_idx,
                         std::vector<float>& out_val);

void gpu_sparse_to_dense(const int* d_idx, const float* d_val, int nnz,
                         float* d_dense, int n);
