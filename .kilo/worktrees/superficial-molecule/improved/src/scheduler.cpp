#include "core/scheduler.h"
#include "cpu/spmspv/kernels.h"
#include "cpu/spmspv/pm_bhash.h"
#include "cpu/spmspv/lb_pm_bhash.h"
#include "cpu/spmspv/pb_mspa.h"
#include "cpu/spmspv/lb_pb_mspa.h"
#include "cpu/spmspv/gustavson.h"
#include "gpu/gpu_kernels.h"
#include <stdexcept>
#include <iostream>

SparseVector dispatch_kernel(
    const std::string& kernel_name,
    const CSRMatrix& A_csr,
    const CSCMatrix& A_csc,
    const SparseVector& x,
    const std::vector<bool>& mask,
    double& out_ms)
{
    KernelResult res_cpu;
    GPUKernelResult res_gpu;

    if (kernel_name == "PM-BHash") {
        res_cpu = cpu_pm_bhash(A_csc, x, mask, -1);
        out_ms = res_cpu.exec_ms;
        return std::move(res_cpu.y);
    } else if (kernel_name == "LB-PM-BHash") {
        res_cpu = cpu_lb_pm_bhash(A_csc, x, mask, -1);
        out_ms = res_cpu.exec_ms;
        return std::move(res_cpu.y);
    } else if (kernel_name == "PB-MSPA") {
        res_cpu = cpu_pb_mspa(A_csc, x, mask, -1);
        out_ms = res_cpu.exec_ms;
        return std::move(res_cpu.y);
    } else if (kernel_name == "LB-PB-MSPA") {
        res_cpu = cpu_lb_pb_mspa(A_csc, x, mask, -1);
        out_ms = res_cpu.exec_ms;
        return std::move(res_cpu.y);
    } else if (kernel_name == "LB-MSPA") {
        res_cpu = cpu_gustavson(A_csc, x, mask);
        out_ms = res_cpu.exec_ms;
        return std::move(res_cpu.y);
    }
    else if (kernel_name == "Sort-Based SpMSpV") {
        res_gpu = gpu_sort_based_spmspv(A_csr, x, mask);
        out_ms = res_gpu.exec_ms;
        return std::move(res_gpu.y);
    } else if (kernel_name == "SpMV") {
        res_gpu = gpu_csr_vector_spmv(A_csr, x, mask);
        out_ms = res_gpu.exec_ms;
        return std::move(res_gpu.y);
    } else if (kernel_name == "Merge-Based SpMV") {
        res_gpu = gpu_merge_spmv(A_csr, x, mask);
        out_ms = res_gpu.exec_ms;
        return std::move(res_gpu.y);
    }

    throw std::runtime_error("Unknown kernel: " + kernel_name);
}
