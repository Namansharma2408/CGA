#pragma once
#include <vector>
#include <string>

struct IterationRecord {
    int depth{0};
    int frontier_nnz{0};
    int next_nnz{0};
    float x_density{0};
    float m_density{0};
    std::string kernel;
    double ms{0};
    double overhead_ms{0};
};

struct BFSResult {
    int source{0};
    int n{0};
    int n_visited{0};
    int max_depth{0};
    double total_ms{0.0};
    std::vector<int> visited;
    std::vector<IterationRecord> iters;

    void print_summary() const;
};
