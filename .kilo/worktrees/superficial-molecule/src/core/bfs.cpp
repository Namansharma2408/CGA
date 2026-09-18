#include "core/bfs.h"
#include "core/scheduler.h"
#include "core/execution_flow.h"
#include "models/inference.h"
#include "models/features.h"
#include <iostream>
#include <chrono>
#include <algorithm>

void BFSResult::print_summary() const {
    std::cout << "\n============================================\n";
    std::cout << "          CGA-BFS Execution Summary        \n";
    std::cout << "============================================\n";
    std::cout << "Source Vertex: " << source << "\n";
    std::cout << "Max Depth: " << max_depth << "\n";
    std::cout << "Vertices Visited: " << n_visited << " / " << n << "\n";
    std::cout << "Total Time: " << total_ms << " ms\n";
    std::cout << "--------------------------------------------\n";
    std::cout << "Iter\tFrntr_NNZ\tKernel\t\tTime(ms)\tOverhead(ms)\n";
    for (const auto& it : iters) {
        std::cout << it.depth << "\t" << it.frontier_nnz << "\t\t"
                  << it.kernel << (it.kernel.size() < 8 ? "\t\t" : "\t")
                  << it.ms << "\t\t" << it.overhead_ms << "\n";
    }
    std::cout << "============================================\n";
}

BFSResult run_bfs(
    const CSRMatrix& A_csr,
    const CSCMatrix& A_csc,
    int source_vertex,
    InferenceEngine::Mode mode)
{
    auto t_start_total = std::chrono::high_resolution_clock::now();
    int n = A_csr.n_rows;
    BFSResult result;
    result.source = source_vertex;
    result.n = n;

    std::vector<bool> mask(n, true);
    mask[source_vertex] = false;
    SparseVector x(n, {source_vertex}, {1.0f});

    InferenceEngine engine;
    engine.mode = mode;

    MatrixStats mstats;
    if (mode != InferenceEngine::BASELINE) {
        mstats = compute_matrix_stats(A_csr.indptr.data(), n, A_csr.nnz);
        if (mode == InferenceEngine::DTA_MODE) engine.init_dta(mstats);
    }

    int depth = 0;
    int m_nnz = A_csr.nnz;

    while (x.nnz > 0) {
        auto t_iter_start = std::chrono::high_resolution_clock::now();
        IterationRecord rec;
        rec.depth = depth;
        rec.frontier_nnz = x.nnz;
        rec.x_density = (float)x.nnz / n;
        rec.m_density = (float)m_nnz / A_csr.nnz;

        std::string platform, kernel;
        bool x_decreasing = (depth > 0 && result.iters.back().frontier_nnz > x.nnz);

        if (mode == InferenceEngine::BASELINE) {
            platform = "CPU"; kernel = "LB-MSPA";
        } else {
            FeatureVector feat = build_feature_vector(mstats, x.nnz, m_nnz,
                depth > 0 ? result.iters.back().frontier_nnz : 0);
            auto sel = determine_execution_flow(engine, feat, x_decreasing);
            platform = sel.first;
            kernel = sel.second;
        }

        rec.kernel = kernel;
        auto t_inf = std::chrono::high_resolution_clock::now();
        rec.overhead_ms = std::chrono::duration<double,std::milli>(t_inf - t_iter_start).count();

        std::cout << "    [DEBUG] Frontier: " << x.nnz << ", Dispatching: [" << kernel << "]" << std::endl;

        double exec_ms = 0;
        SparseVector x_next = dispatch_kernel(kernel, A_csr, A_csc, x, mask, exec_ms);
        rec.ms = exec_ms;

        if (mode == InferenceEngine::DTA_MODE) {
            FeatureVector feat = build_feature_vector(mstats, x.nnz, m_nnz,
                depth > 0 ? result.iters.back().frontier_nnz : 0);
            engine.update_dta(kernel, exec_ms, feat);
        }

        for (int i = 0; i < x_next.nnz; ++i) {
            int v = x_next.indices[i];
            if (mask[v]) {
                mask[v] = false;
                result.visited.push_back(v);
            }
        }

        m_nnz = std::max(0, m_nnz - (int)(x.nnz * mstats.avg_degree));
        x = std::move(x_next);
        result.iters.push_back(rec);
        depth++;
    }

    result.max_depth = depth;
    result.n_visited = (int)result.visited.size() + 1;
    auto t_total_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double,std::milli>(t_total_end - t_start_total).count();
    return result;
}
