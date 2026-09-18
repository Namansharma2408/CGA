#include "gpu/gpu_kernels.h"
#include "gpu/utils/memory.h"
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

__device__ int2 merge_path_search(
    int diag, const int* __restrict__ indptr, int n_rows, int n_nnz)
{
    int lo = max(0, diag - n_nnz);
    int hi = min(diag, n_rows);
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (indptr[mid + 1] <= diag - mid) lo = mid + 1;
        else                               hi = mid;
    }
    return make_int2(lo, diag - lo);
}

__global__ void merge_spmv_kernel(
    const int*   __restrict__ indptr,
    const int*   __restrict__ indices,
    const float* __restrict__ data,
    const float* __restrict__ x_dense,
    const int*   __restrict__ mask_int,
    float*       __restrict__ y_dense,
    float*       __restrict__ carry_val,
    int*         __restrict__ carry_row,
    int n_rows, int n_nnz, int items_per_block)
{
    extern __shared__ float smem[];

    int tile_start = blockIdx.x * items_per_block;
    int tile_end   = min(tile_start + items_per_block, n_rows + n_nnz);

    int2 start = merge_path_search(tile_start, indptr, n_rows, n_nnz);

    int row0 = start.x;
    float running = 0.0f;
    int   cur_row = row0;

    for (int item = tile_start + threadIdx.x; item < tile_end; item += blockDim.x) {
        int2 coord = merge_path_search(item, indptr, n_rows, n_nnz);
        int r = coord.x, p = coord.y;
        if (r == cur_row && p < n_nnz) {
            running += data[p] * x_dense[indices[p]];
        } else {
            if (mask_int[cur_row]) y_dense[cur_row] = running;
            else                   y_dense[cur_row] = 0.0f;
            cur_row = r;
            running = (p < n_nnz) ? data[p] * x_dense[indices[p]] : 0.0f;
        }
    }
    if (threadIdx.x == blockDim.x - 1) {
        carry_val[blockIdx.x] = running;
        carry_row[blockIdx.x] = cur_row;
    }
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x > 0) {
        if (carry_row[blockIdx.x-1] == cur_row)
            atomicAdd(&y_dense[cur_row], carry_val[blockIdx.x-1]);
    }
}

GPUKernelResult gpu_merge_spmv(
    const CSRMatrix& A, const SparseVector& x,
    const std::vector<bool>& mask)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    int n = A.n_rows, nnz = A.nnz;

    std::vector<float> x_dense(n, 0.0f);
    for (int k = 0; k < x.nnz; ++k) x_dense[x.indices[k]] = x.values[k];
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

    int threads = 128, items_per_block = threads * 2;
    int blocks  = ((n + nnz) + items_per_block - 1) / items_per_block;
    DevPtr<float> d_carry_val(blocks); DevPtr<int> d_carry_row(blocks);

    merge_spmv_kernel<<<blocks, threads, threads*sizeof(float)>>>(
        d_indptr.get(), d_indices.get(), d_data.get(),
        d_x.get(), d_mask.get(), d_y.get(),
        d_carry_val.get(), d_carry_row.get(),
        n, nnz, items_per_block);
    cudaDeviceSynchronize();

    std::vector<float> h_y(n);
    d_y.download(h_y.data());

    std::vector<int>   out_idx; std::vector<float> out_val;
    for (int i = 0; i < n; ++i)
        if (mask[i] && h_y[i] != 0.0f) {
            out_idx.push_back(i); out_val.push_back(h_y[i]);
        }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    return GPUKernelResult{SparseVector(n, std::move(out_idx), std::move(out_val)), ms};
}
