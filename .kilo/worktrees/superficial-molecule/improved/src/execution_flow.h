#pragma once
#include "models/inference.h"
#include "models/features.h"

std::pair<std::string, std::string> determine_execution_flow(
    InferenceEngine& engine,
    const FeatureVector& feat,
    bool x_decreasing);
