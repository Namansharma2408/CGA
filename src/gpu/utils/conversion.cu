#include "gpu/utils/conversion.h"
#include <cuda_runtime.h>
#include <vector>

int gpu_dense_to_sparse(const float* d_dense, int n,
                        std::vector<int>& out_idx, std::vector<float>& out_val)
{
    // Simple CPU-based conversion since Thrust has CUDA API compatibility issues
    // This is a fallback that works without Thrust
    out_idx.clear();
    out_val.clear();
    for (int i = 0; i < n; ++i) {
        if (d_dense[i] != 0.0f) {
            out_idx.push_back(i);
            out_val.push_back(d_dense[i]);
        }
    }
    return (int)out_idx.size();
}

__global__ void scatter_kernel(const int* idx, const float* val, int nnz, float* dst) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nnz) dst[idx[k]] = val[k];
}

void gpu_sparse_to_dense(const int* d_idx, const float* d_val, int nnz,
                         float* d_dense, int n)
{
    cudaMemset(d_dense, 0, n * sizeof(float));
    if (nnz == 0) return;
    int threads = 256, blocks = (nnz + threads - 1) / threads;
    scatter_kernel<<<blocks, threads>>>(d_idx, d_val, nnz, d_dense);
    cudaDeviceSynchronize();
}