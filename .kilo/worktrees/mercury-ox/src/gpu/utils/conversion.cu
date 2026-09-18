
#include "gpu/utils/conversion.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <vector>

int gpu_dense_to_sparse(const float* d_dense, int n,
                        std::vector<int>& out_idx, std::vector<float>& out_val)
{
    thrust::device_ptr<const float> dp(d_dense);
    thrust::device_vector<int>   d_idx(n);
    thrust::device_vector<float> d_val(n);
    auto it = thrust::make_counting_iterator(0);
    auto end = thrust::copy_if(
        thrust::make_zip_iterator(thrust::make_tuple(it, dp)),
        thrust::make_zip_iterator(thrust::make_tuple(it+n, dp+n)),
        dp,
        thrust::make_zip_iterator(thrust::make_tuple(d_idx.begin(), d_val.begin())),
        [] __device__ (float v) { return v != 0.0f; }
    );
    int nnz = (int)(end - thrust::make_zip_iterator(
                       thrust::make_tuple(d_idx.begin(), d_val.begin())));
    out_idx.resize(nnz);
    out_val.resize(nnz);
    thrust::copy(d_idx.begin(), d_idx.begin() + nnz, out_idx.begin());
    thrust::copy(d_val.begin(), d_val.begin() + nnz, out_val.begin());
    return nnz;
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
