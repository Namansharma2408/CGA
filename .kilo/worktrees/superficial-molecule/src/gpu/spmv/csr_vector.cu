
#include "gpu/gpu_kernels.h"
#include "gpu/utils/memory.h"
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

#define WARP_SIZE 32

__global__ void csr_vector_spmv_kernel(
    const int*   __restrict__ indptr,
    const int*   __restrict__ indices,
    const float* __restrict__ data,
    const float* __restrict__ x_dense,
    const int*   __restrict__ mask_int,
    float* __restrict__ y_dense,
    int n_rows)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;
    if (warp_id >= n_rows) return;

    int row = warp_id;
    if (!mask_int[row]) { y_dense[row] = 0.0f; return; }

    float sum  = 0.0f;
    int start = indptr[row];
    int end   = indptr[row + 1];

    for (int p = start + lane; p < end; p += WARP_SIZE)
        sum += data[p] * x_dense[indices[p]];

    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) y_dense[row] = sum;
}

GPUKernelResult gpu_csr_vector_spmv(
    const CSRMatrix& A, const SparseVector& x,
    const std::vector<bool>& mask)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    int n   = A.n_rows, nnz = A.nnz;

    std::vector<float> x_dense(n, 0.0f);
    for (int k = 0; k < x.nnz; ++k)
        x_dense[x.indices[k]] = x.values[k];
    std::vector<int> mask_int(n);
    for (int i = 0; i < n; ++i) mask_int[i] = mask[i] ? 1 : 0;

    DevPtr<int>   d_indptr(n+1), d_indices(nnz), d_mask(n);
    DevPtr<float> d_data(nnz), d_x(n), d_y(n);

    d_indptr.upload(A.indptr.data());
    d_indices.upload(A.indices.data());
    d_data.upload(A.data.data());
    d_x.upload(x_dense.data());
    d_mask.upload(mask_int.data());
    gpu_memset_zero(d_y.get(), n * sizeof(float));

    int warps_per_block = 256 / WARP_SIZE;
    int blocks = (n + warps_per_block - 1) / warps_per_block;
    csr_vector_spmv_kernel<<<blocks, 256>>>(
        d_indptr.get(), d_indices.get(), d_data.get(),
        d_x.get(), d_mask.get(), d_y.get(), n);
    cudaDeviceSynchronize();

    std::vector<float> h_y(n);
    d_y.download(h_y.data());

    std::vector<int>   out_idx; out_idx.reserve(4096);
    std::vector<float> out_val; out_val.reserve(4096);
    for (int i = 0; i < n; ++i)
        if (mask[i] && h_y[i] != 0.0f) {
            out_idx.push_back(i);
            out_val.push_back(h_y[i]);
        }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    return GPUKernelResult{SparseVector(n, std::move(out_idx), std::move(out_val)), ms};
}
