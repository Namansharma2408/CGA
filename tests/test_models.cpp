#include "models/inference.h"
#include "core/kernel_registry.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Running ML Model Inference Unit Tests...\n";
    InferenceEngine engine;

    // Test 1: dense + scale-free -> GPU SpMV (Model1 GPU branch).
    {
        FeatureVector feat;
        feat[F_X_DENSITY] = 0.5f;
        feat[F_IS_SCALE_FREE] = 1.0f;
        auto sel = engine.select_kernel(feat, false);
        std::cout << "Test 1 -> " << sel.first << " : " << sel.second << "\n";
        assert(sel.first == "GPU" && sel.second == "SpMV");
    }
    // Test 2: sparse non-scale-free + decreasing -> CPU PM-BHash (Model4 CPU path).
    {
        FeatureVector feat;
        feat[F_X_DENSITY] = 0.01f;
        feat[F_IS_SCALE_FREE] = 0.0f;
        auto sel = engine.select_kernel(feat, true);
        std::cout << "Test 2 -> " << sel.first << " : " << sel.second << "\n";
        assert(sel.first == "CPU" && sel.second == "PM-BHash");
    }
    // P0-6: Model1 branch coverage — scale-free but sparse -> CPU (was GPU bug).
    {
        auto m1 = create_model1_platform();
        FeatureVector feat;
        feat[F_IS_SCALE_FREE] = 1.0f;
        feat[F_X_DENSITY] = 0.01f;  // sparse
        std::string p = m1->predict_name(feat);
        std::cout << "Test 3 (M1 sparse scale-free) -> " << p << "\n";
        assert(p == "CPU");
        feat[F_X_DENSITY] = 0.5f;  // dense
        p = m1->predict_name(feat);
        std::cout << "Test 4 (M1 dense scale-free) -> " << p << "\n";
        assert(p == "GPU");
        feat[F_IS_SCALE_FREE] = 0.0f;
        feat[F_X_DENSITY] = 0.9f;
        p = m1->predict_name(feat);
        std::cout << "Test 5 (M1 non-scale-free) -> " << p << "\n";
        assert(p == "CPU");
    }
    // P0-5: every canonical kernel dispatches (registry validation).
    for (const auto& k : all_kernel_names()) {
        KernelId id = kernel_id_from_string(k);  // throws on unknown
        (void)id;
    }
    // Legacy aliases still resolve (backward compat).
    assert(kernel_id_from_string("Gustavson") == KernelId::LB_MSPA);
    assert(kernel_id_from_string("Sort-Based") == KernelId::SORT_SPMSPV);
    assert(kernel_id_from_string("CSR-Vector") == KernelId::SPMV);
    assert(kernel_id_from_string("Merge-Based") == KernelId::MERGE_SPMV);

    // P1-1: RAII — init_dta creates DTA only in DTA_MODE.
    {
        InferenceEngine e2;
        e2.mode = InferenceEngine::STATIC;
        MatrixStats ms;
        ms.n = 10; ms.nnz = 20;
        e2.init_dta(ms);
        assert(!e2.has_dta());
        e2.mode = InferenceEngine::DTA_MODE;
        e2.init_dta(ms);
        assert(e2.has_dta());
    }

    std::cout << "Models Integration tests passed.\n";
    return 0;
}
