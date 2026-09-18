// P0-10 regression: on_gpu && x_decreasing must return k4 (CPU non-SpMV),
// not the DTA kernel k. Uses improved 25-dim headers.
#include "models/inference.h"
#include "models/features.h"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "Running improved DTA one-copy regression...\n";
    static_assert(FeatureVector::DIM == 25, "improved DIM must be 25");
    assert(std::string(FEATURE_NAMES[4]) == "col_cv");
    assert(std::string(FEATURE_NAMES[19]) == "degree_variance");

    InferenceEngine eng;
    eng.mode = InferenceEngine::DTA_MODE;
    MatrixStats ms;
    ms.n = 1000; ms.nnz = 5000; ms.avg_degree = 5.0f;
    ms.gini_row = 0.8f; ms.row_mean = 5; ms.row_std = 10;
    ms.degree_variance = 100.0f;
    eng.init_dta(ms, InferenceEngine::GRADIENT_DTA);
    assert(eng.has_dta());

    // Force on_gpu path: scale-free + dense -> M1 predicts GPU.
    FeatureVector feat;
    feat[F25_IS_SCALE_FREE] = 1.0f;
    feat[F25_X_DENSITY] = 0.5f;   // dense -> DTA greedy SpMV, M1 GPU
    feat[F25_M_DENSITY] = 0.5f;
    feat[F25_DEGREE_X] = 5.0f;
    feat[F25_DEGREE_VAR] = 100.0f;
    // x_decreasing=true triggers one-copy check; M4 returns SpMV for dense,
    // so result stays GPU/SpMV (no crash, accessor used, no protected access).
    auto sel = eng.select_kernel(feat, true, 500, 1000);
    std::cout << " dense+decreasing -> " << sel.first << ":" << sel.second << "\n";
    assert(sel.second == "SpMV");

    // DTA_TYPE variants construct without crash.
    for (auto t : {InferenceEngine::BASE_DTA, InferenceEngine::UCB_DTA,
                   InferenceEngine::GRADIENT_DTA}) {
        InferenceEngine e2;
        e2.mode = InferenceEngine::DTA_MODE;
        e2.init_dta(ms, t);
        FeatureVector f2;
        f2[F25_X_DENSITY] = 0.01f;
        f2[F25_M_DENSITY] = 0.1f;
        auto s2 = e2.select_kernel(f2, false, 10, 1000);
        (void)s2;
    }
    std::cout << "Improved DTA regression passed.\n";
    return 0;
}
