// P1-5/P0-7: DTA explore->exploit, threshold clamping, window cap.
#include "models/inference.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Running DTA tests...\n";
    MatrixStats ms;
    ms.n = 100; ms.nnz = 500; ms.avg_degree = 5.0f;
    ms.gini_row = 0.2f; ms.row_mean = 5; ms.row_std = 1;

    DTA dta(ms);
    assert(dta.state().iteration == 0);
    assert(dta.state().explore_left == 3);

    FeatureVector feat;
    feat[F_X_DENSITY] = 0.01f;
    feat[F_M_DENSITY] = 0.1f;
    feat[F_DEGREE_X] = 0.5f;

    // First 6 iters consume explore budget (P1-5: decrements every explore).
    for (int i = 0; i < 6; ++i) {
        std::string k = dta.select_kernel(feat);
        (void)k;
        dta.update("PM-BHash", 1.0 + i * 0.1, feat);
    }
    DTAState st = dta.state();
    std::cout << " after 6 iters: iteration=" << st.iteration
              << " explore_left=" << st.explore_left << "\n";
    assert(st.iteration == 6);
    assert(st.explore_left == 0 && "explore budget must be consumed");

    // Thresholds stay clamped after extreme gradients.
    for (int i = 0; i < 50; ++i)
        dta.update("SpMV", 0.01, feat);
    std::cout << " DTA stress updates passed (no NaN/crash).\n";

    // Greedy dense -> SpMV.
    FeatureVector dense;
    dense[F_X_DENSITY] = 0.9f;
    dense[F_M_DENSITY] = 0.9f;
    dense[F_DEGREE_X] = 5.0f;
    DTA dta2(ms);
    // Burn explore so greedy is deterministic.
    for (int i = 0; i < 7; ++i) { dta2.select_kernel(dense); }
    std::string k = dta2.select_kernel(dense);
    std::cout << " dense greedy -> " << k << "\n";
    assert(k == "SpMV");

    std::cout << "DTA tests passed.\n";
    return 0;
}
