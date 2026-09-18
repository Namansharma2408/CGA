
#include "gpu/gpu_kernels.h"
#include "gpu/utils/memory.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <chrono>
#include <vector>

__global__ void gather_kernel(
    const int* __restrict__ csc_indptr,
    const int* __restrict__ csc_indices,
    const float* __restrict__ csc_data,
    const int*   __restrict__ x_idx,
    const float* __restrict__ x_val,
    int x_nnz,
    const int* __restrict__ col_offsets,
    int* d_pair_row,
    float* d_pair_val)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= x_nnz) return;
    int j    = x_idx[k];
    float xj = x_val[k];
    int start = csc_indptr[j];
    int end   = csc_indptr[j + 1];
    int out_base = col_offsets[k];
    for (int p = start; p < end; ++p) {
        d_pair_row[out_base + (p - start)] = csc_indices[p];
        d_pair_val[out_base + (p - start)] = csc_data[p] * xj;
    }
}

GPUKernelResult gpu_sort_based_spmspv(
    const CSRMatrix& A_csr,
    const SparseVector& x,
    const std::vector<bool>& mask)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int n   = A_csr.n_rows;
    int nnz = A_csr.nnz;
    int x_nnz = x.nnz;

    std::vector<int>   csc_indptr(n + 1, 0);
    std::vector<int>   csc_indices(nnz);
    std::vector<float> csc_data(nnz);
    for (int k = 0; k < nnz; ++k)
        csc_indptr[A_csr.indices[k] + 1]++;
    for (int j = 0; j < n; ++j)
        csc_indptr[j+1] += csc_indptr[j];
    {
        std::vector<int> pos(csc_indptr.begin(), csc_indptr.begin()+n);
        for (int i = 0; i < n; ++i)
            for (int k = A_csr.indptr[i]; k < A_csr.indptr[i+1]; ++k) {
                int j = A_csr.indices[k];
                int p = pos[j]++;
                csc_indices[p] = i;
                csc_data[p]    = A_csr.data[k];
            }
    }

    std::vector<int> col_nnz(x_nnz), col_off(x_nnz+1, 0);
    for (int k = 0; k < x_nnz; ++k)
        col_nnz[k] = csc_indptr[x.indices[k]+1] - csc_indptr[x.indices[k]];
    for (int k = 0; k < x_nnz; ++k) col_off[k+1] = col_off[k] + col_nnz[k];
    int total_pairs = col_off[x_nnz];

    if (total_pairs == 0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        return GPUKernelResult{SparseVector(n),
            std::chrono::duration<double,std::milli>(t1-t0).count()};
    }

    DevPtr<int>   d_csc_indptr(n+1),  d_csc_indices(nnz),
                  d_x_idx(x_nnz),     d_col_off(x_nnz+1),
                  d_pair_row(total_pairs);
    DevPtr<float> d_csc_data(nnz),    d_x_val(x_nnz),
                  d_pair_val(total_pairs);

    d_csc_indptr.upload(csc_indptr.data());
    d_csc_indices.upload(csc_indices.data());
    d_csc_data.upload(csc_data.data());
    d_x_idx.upload(x.indices.data());
    d_x_val.upload(x.values.data());
    d_col_off.upload(col_off.data());

    {
        int threads = 256, blocks = (x_nnz + threads - 1) / threads;
        gather_kernel<<<blocks, threads>>>(
            d_csc_indptr.get(), d_csc_indices.get(), d_csc_data.get(),
            d_x_idx.get(), d_x_val.get(), x_nnz,
            d_col_off.get(), d_pair_row.get(), d_pair_val.get());
        cudaDeviceSynchronize();
    }

    thrust::device_ptr<int>   tp_row(d_pair_row.get());
    thrust::device_ptr<float> tp_val(d_pair_val.get());
    thrust::sort_by_key(tp_row, tp_row + total_pairs, tp_val);

    thrust::device_vector<int>   d_out_keys(total_pairs);
    thrust::device_vector<float> d_out_vals(total_pairs);
    auto end_it = thrust::reduce_by_key(
        tp_row, tp_row + total_pairs, tp_val,
        d_out_keys.begin(), d_out_vals.begin());
    int out_nnz = (int)(end_it.first - d_out_keys.begin());

    std::vector<int>   h_rows(out_nnz);
    std::vector<float> h_vals(out_nnz);
    thrust::copy(d_out_keys.begin(), d_out_keys.begin()+out_nnz, h_rows.begin());
    thrust::copy(d_out_vals.begin(), d_out_vals.begin()+out_nnz, h_vals.begin());

    std::vector<int>   out_idx; out_idx.reserve(out_nnz);
    std::vector<float> out_val; out_val.reserve(out_nnz);
    for (int k = 0; k < out_nnz; ++k) {
        if (mask[h_rows[k]]) {
            out_idx.push_back(h_rows[k]);
            out_val.push_back(h_vals[k]);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    return GPUKernelResult{SparseVector(n, std::move(out_idx), std::move(out_val)), ms};
}
