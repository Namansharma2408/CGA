#include "core/execution_flow.h"
#include "models/inference.h"

bool model0_prefilter(const FeatureVector& feat, int x_nnz, int n,
                      std::pair<std::string, std::string>& out) {
    (void)feat;
    if (x_nnz == 1) { out = {"CPU", "PM-BHash"}; return true; }
    if (n > 0 && x_nnz > static_cast<int>(n * 0.30f)) {
        out = {"GPU", "SpMV"};
        return true;
    }
    return false;
}

std::pair<std::string, std::string> determine_execution_flow(
    InferenceEngine& engine,
    const FeatureVector& feat,
    bool x_decreasing,
    int x_nnz,
    int n)
{
    std::pair<std::string, std::string> pre;
    if (model0_prefilter(feat, x_nnz, n, pre)) return pre;
    return engine.select_kernel(feat, x_decreasing);
}
