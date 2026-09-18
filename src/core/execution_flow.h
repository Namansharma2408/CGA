#pragma once
#include "models/inference.h"
#include "models/features.h"

// P1-4: Model-0 prefilter + one-copy policy live here so bfs.cpp stays
// orchestration-only. Returns {platform, kernel}.
std::pair<std::string, std::string> determine_execution_flow(
    InferenceEngine& engine,
    const FeatureVector& feat,
    bool x_decreasing,
    int x_nnz = 0,
    int n = 0);

// Model-0: frontier==1 -> PM-BHash; dense>30% -> SpMV. Returns true if handled.
bool model0_prefilter(const FeatureVector& feat, int x_nnz, int n,
                      std::pair<std::string, std::string>& out);
