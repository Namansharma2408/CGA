#pragma once
// Canonical kernel ontology (P0-5, P1-2). Single source of truth for
// C++ dispatch, DTA candidates, and Python training labels (see KERNELS.md).
#include <stdexcept>
#include <string>
#include <vector>

enum class KernelId {
    PM_BHASH = 0,
    LB_PM_BHASH = 1,
    PB_MSPA = 2,
    LB_PB_MSPA = 3,
    LB_MSPA = 4,          // CPU baseline; implemented via cpu_gustavson()
    SORT_SPMSPV = 5,      // GPU Sort-Based SpMSpV
    SPMV = 6,             // GPU CSR-Vector SpMV
    MERGE_SPMV = 7        // GPU Merge-Based SpMV
};

inline std::string kernel_id_to_string(KernelId id) {
    switch (id) {
        case KernelId::PM_BHASH: return "PM-BHash";
        case KernelId::LB_PM_BHASH: return "LB-PM-BHash";
        case KernelId::PB_MSPA: return "PB-MSPA";
        case KernelId::LB_PB_MSPA: return "LB-PB-MSPA";
        case KernelId::LB_MSPA: return "LB-MSPA";
        case KernelId::SORT_SPMSPV: return "Sort-Based SpMSpV";
        case KernelId::SPMV: return "SpMV";
        case KernelId::MERGE_SPMV: return "Merge-Based SpMV";
    }
    return "UNKNOWN";
}

// Accepts canonical names + legacy aliases from training scripts.
// Legacy: "Gustavson" == "LB-MSPA" (same kernel, old short name);
//         "Sort-Based" == "Sort-Based SpMSpV", "CSR-Vector" == "SpMV",
//         "Merge-Based" == "Merge-Based SpMV".
inline KernelId kernel_id_from_string(const std::string& name) {
    if (name == "PM-BHash") return KernelId::PM_BHASH;
    if (name == "LB-PM-BHash") return KernelId::LB_PM_BHASH;
    if (name == "PB-MSPA") return KernelId::PB_MSPA;
    if (name == "LB-PB-MSPA") return KernelId::LB_PB_MSPA;
    if (name == "LB-MSPA" || name == "Gustavson") return KernelId::LB_MSPA;
    if (name == "Sort-Based SpMSpV" || name == "Sort-Based") return KernelId::SORT_SPMSPV;
    if (name == "SpMV" || name == "CSR-Vector") return KernelId::SPMV;
    if (name == "Merge-Based SpMV" || name == "Merge-Based") return KernelId::MERGE_SPMV;
    throw std::runtime_error("Unknown kernel: " + name +
        " (valid: PM-BHash, LB-PM-BHash, PB-MSPA, LB-PB-MSPA, LB-MSPA/Gustavson, "
        "Sort-Based SpMSpV, SpMV, Merge-Based SpMV)");
}

inline std::vector<std::string> all_kernel_names() {
    return {"PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA",
            "LB-MSPA", "Sort-Based SpMSpV", "SpMV", "Merge-Based SpMV"};
}

inline bool is_gpu_kernel_name(const std::string& name) {
    return name == "Sort-Based SpMSpV" || name == "Sort-Based" ||
           name == "SpMV" || name == "CSR-Vector" ||
           name == "Merge-Based SpMV" || name == "Merge-Based";
}
