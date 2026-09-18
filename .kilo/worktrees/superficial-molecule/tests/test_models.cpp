#include "models/inference.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Running ML Model Inference Unit Tests...\n";
    InferenceEngine engine;

    FeatureVector feat;
    feat.v[10] = 0.5f;
    feat.v[8] = 1.0f;

    auto sel1 = engine.select_kernel(feat, false);
    std::cout << "Test 1 -> " << sel1.first << " : " << sel1.second << "\n";
    assert(sel1.first == "GPU" && sel1.second == "SpMV");

    feat.v[10] = 0.01f;
    feat.v[8] = 0.0f;
    auto sel2 = engine.select_kernel(feat, true);
    std::cout << "Test 2 -> " << sel2.first << " : " << sel2.second << "\n";
    assert(sel2.first == "CPU" && sel2.second == "PM-BHash");

    std::cout << "Models Integration tests passed dynamically.\n";
    return 0;
}
