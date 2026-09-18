#include "core/execution_flow.h"
#include "models/inference.h"

std::pair<std::string, std::string> determine_execution_flow(
    InferenceEngine& engine,
    const FeatureVector& feat,
    bool x_decreasing)
{
    return engine.select_kernel(feat, x_decreasing);
}
