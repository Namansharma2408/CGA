#include "core/scheduler.h"
#include "core/kernel_registry.h"
#include "cpu/spmspv/kernels.h"
#include "cpu/spmspv/pm_bhash.h"
#include "cpu/spmspv/lb_pm_bhash.h"
#include "cpu/spmspv/pb_mspa.h"
#include "cpu/spmspv/lb_pb_mspa.h"
#include "cpu/spmspv/gustavson.h"
#include "gpu/gpu_kernels.h"
#include <functional>
#include <map>
#include <stdexcept>
#include <iostream>

// P1-2: OCP registry — adding a kernel = one map entry, no if-else chain edit.
using CpuHandler = std::function<KernelResult(const CSCMatrix&, const SparseVector&, const std::vector<bool>&)>;
using GpuHandler = std::function<GPUKernelResult(const CSRMatrix&, const SparseVector&, const std::vector<bool>&)>;

static const std::map<KernelId, CpuHandler>& cpu_registry() {
    static const std::map<KernelId, CpuHandler> m = {
        {KernelId::PM_BHASH, [](const CSCMatrix& a, const SparseVector& x, const std::vector<bool>& mk){ return cpu_pm_bhash(a, x, mk, -1); }},
        {KernelId::LB_PM_BHASH, [](const CSCMatrix& a, const SparseVector& x, const std::vector<bool>& mk){ return cpu_lb_pm_bhash(a, x, mk, -1); }},
        {KernelId::PB_MSPA, [](const CSCMatrix& a, const SparseVector& x, const std::vector<bool>& mk){ return cpu_pb_mspa(a, x, mk, -1); }},
        {KernelId::LB_PB_MSPA, [](const CSCMatrix& a, const SparseVector& x, const std::vector<bool>& mk){ return cpu_lb_pb_mspa(a, x, mk, -1); }},
        // LB-MSPA baseline is implemented via Gustavson SpMSpV (see KERNELS.md).
        {KernelId::LB_MSPA, [](const CSCMatrix& a, const SparseVector& x, const std::vector<bool>& mk){ return cpu_gustavson(a, x, mk); }},
    };
    return m;
}

static const std::map<KernelId, GpuHandler>& gpu_registry() {
    static const std::map<KernelId, GpuHandler> m = {
        {KernelId::SORT_SPMSPV, [](const CSRMatrix& a, const SparseVector& x, const std::vector<bool>& mk){ return gpu_sort_based_spmspv(a, x, mk); }},
        {KernelId::SPMV, [](const CSRMatrix& a, const SparseVector& x, const std::vector<bool>& mk){ return gpu_csr_vector_spmv(a, x, mk); }},
        {KernelId::MERGE_SPMV, [](const CSRMatrix& a, const SparseVector& x, const std::vector<bool>& mk){ return gpu_merge_spmv(a, x, mk); }},
    };
    return m;
}

SparseVector dispatch_kernel(
    const std::string& kernel_name,
    const CSRMatrix& A_csr,
    const CSCMatrix& A_csc,
    const SparseVector& x,
    const std::vector<bool>& mask,
    double& out_ms)
{
    // Validates name (throws with valid-ID list on unknown, incl. legacy aliases).
    KernelId id = kernel_id_from_string(kernel_name);
    auto cit = cpu_registry().find(id);
    if (cit != cpu_registry().end()) {
        KernelResult r = cit->second(A_csc, x, mask);
        out_ms = r.exec_ms;
        return std::move(r.y);
    }
    auto git = gpu_registry().find(id);
    if (git != gpu_registry().end()) {
        GPUKernelResult r = git->second(A_csr, x, mask);
        out_ms = r.exec_ms;
        return std::move(r.y);
    }
    throw std::runtime_error("No handler registered for kernel: " + kernel_name);
}
