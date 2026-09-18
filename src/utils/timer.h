#pragma once
#include <chrono>

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;

    void start();
    double stopMs();
};
