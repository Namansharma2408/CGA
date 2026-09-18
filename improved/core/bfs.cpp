#include "core/bfs.h"
#include "core/scheduler.h"
#include "core/execution_flow.h"
#include "models/inference.h"
#include "models/features.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <stdexcept>

void BFSResult::print_summary() const {
    std::cout << "\n============================================\n";
    std::cout << "          CGA-BFS Execution Summary        \n";
    std::cout << "============================================\n";
    std::cout << "Source Vertex: " << source << "\n";
    std::cout << "Max Depth:     " << max_depth << "\n";
    std::cout << "Vertices:      " << n_visited << " / " << n << "\n";
    std::cout << "Total Time:    " << total_ms << " ms\n";
    std::cout << "--------------------------------------------\n";
    std::cout << "Iter\tFrntr\tKernel\t\tTime(ms)\tOverhead(ms)\n";
    for (const auto& it : iters) {
        std::cout << it.depth << "\t" << it.frontier_nnz << "\t\t"
                  << it.kernel
                  << (it.kernel.size() < 8 ? "\t\t" : "\t")
                  << it.ms << "\t\t" << it.overhead_ms << "\n";
    }
    std::cout << "============================================\n";
}

namespace { bool g_verbose_impr = false; }
void set_bfs_verbose(bool v) { g_verbose_impr = v; }

BFSResult run_bfs(
    const CSRMatrix& A_csr,
    const CSCMatrix& A_csc,
    int source_vertex,
    InferenceEngine::Mode mode,
    InferenceEngine::DTA_TYPE dta_type)
{
    auto t_start_total = std::chrono::high_resolution_clock::now();
    int n = A_csr.n_rows;
    if (n <= 0) throw std::invalid_argument("empty graph (n<=0)");
    if (source_vertex < 0 || source_vertex >= n)
        throw std::out_of_range("source vertex out of range");
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
        mstats = compute_matrix_stats(
            A_csr.indptr.data(),
            A_csc.indptr.data(),
            n, A_csr.nnz);
        if (mode == InferenceEngine::DTA_MODE)
            engine.init_dta(mstats, dta_type);
    }

    int depth = 0;
    int m_nnz = A_csr.nnz;

    while (x.nnz > 0) {
        auto t_iter_start = std::chrono::high_resolution_clock::now();
        IterationRecord rec;
        rec.depth        = depth;
        rec.frontier_nnz = x.nnz;
        rec.x_density    = n > 0 ? (float)x.nnz / n : 0.0f;
        rec.m_density    = A_csr.nnz > 0 ? (float)m_nnz / A_csr.nnz : 0.0f;

        bool x_decreasing = (depth > 0 &&
                             result.iters.back().frontier_nnz > x.nnz);

        FeatureVector feat;
        bool have_feat = (mode != InferenceEngine::BASELINE);
        if (have_feat) {
            feat = build_feature_vector(
                mstats, x.nnz, m_nnz,
                depth > 0 ? result.iters.back().frontier_nnz : 0);
        }

        std::string platform, kernel;
        if (mode == InferenceEngine::BASELINE) {
            platform = "CPU"; kernel = "LB-MSPA";
        } else {
            auto sel = engine.select_kernel(feat, x_decreasing, x.nnz, n);
            platform = sel.first;
            kernel   = sel.second;
        }

        rec.kernel = kernel;
        auto t_inf = std::chrono::high_resolution_clock::now();
        rec.overhead_ms = std::chrono::duration<double, std::milli>(
            t_inf - t_iter_start).count();

        double exec_ms = 0;
        SparseVector x_next = dispatch_kernel(
            kernel, A_csr, A_csc, x, mask, exec_ms);
        rec.ms = exec_ms;

        if (mode == InferenceEngine::DTA_MODE && have_feat) {
            engine.update_dta(kernel, exec_ms, feat);
        }

        for (int i = 0; i < x_next.nnz; ++i) {
            int v = x_next.indices[i];
            if (v < 0 || v >= n) throw std::out_of_range("kernel OOB vertex");
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

    result.max_depth  = depth;
    result.n_visited  = (int)result.visited.size() + 1;
    auto t_total_end  = std::chrono::high_resolution_clock::now();
    result.total_ms   = std::chrono::duration<double, std::milli>(
        t_total_end - t_start_total).count();
    return result;
}
