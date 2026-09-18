// P0-5: every training label (canonical + legacy) must dispatch.
#include "core/kernel_registry.h"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "Running kernel registry tests...\n";
    // Canonical 8.
    for (const auto& k : all_kernel_names()) {
        KernelId id = kernel_id_from_string(k);
        std::string back = kernel_id_to_string(id);
        assert(back == k && "round-trip must preserve canonical name");
    }
    // Legacy training aliases.
    assert(kernel_id_from_string("Gustavson") == KernelId::LB_MSPA);
    assert(kernel_id_from_string("Sort-Based") == KernelId::SORT_SPMSPV);
    assert(kernel_id_from_string("CSR-Vector") == KernelId::SPMV);
    assert(kernel_id_from_string("Merge-Based") == KernelId::MERGE_SPMV);
    // GPU predicate.
    assert(is_gpu_kernel_name("SpMV"));
    assert(!is_gpu_kernel_name("PM-BHash"));
    // Unknown throws with helpful message.
    bool threw = false;
    try { kernel_id_from_string("Nope-Kernel"); } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("PM-BHash") != std::string::npos);
    }
    assert(threw);
    std::cout << "Kernel registry tests passed.\n";
    return 0;
}
