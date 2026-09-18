// P0-7: BFS golden tests vs reference BFS (line, star, disconnected, self-loop).
#include "core/bfs.h"
#include "matrix/loader.h"
#include "models/inference.h"

BFSResult run_bfs(const CSRMatrix& A_csr, const CSCMatrix& A_csc,
                  int source_vertex, InferenceEngine::Mode mode);
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <utility>
#include <vector>

static void write_mtx(const std::string& path, int n,
                      const std::vector<std::pair<int,int>>& edges_1idx) {
    std::ofstream f(path);
    f << "%%MatrixMarket matrix coordinate pattern symmetric\n";
    f << n << " " << n << " " << edges_1idx.size() << "\n";
    for (auto& [u, v] : edges_1idx) f << u << " " << v << "\n";
}

static std::set<int> ref_bfs(int n,
    const std::vector<std::pair<int,int>>& edges0, int src) {
    std::vector<std::vector<int>> adj(n);
    for (auto& [u, v] : edges0) {
        adj[u].push_back(v);
        if (u != v) adj[v].push_back(u);
    }
    std::set<int> vis{src};
    std::queue<int> q;
    q.push(src);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) if (!vis.count(v)) { vis.insert(v); q.push(v); }
    }
    return vis;
}

static void check(const std::string& name, int n,
                  const std::vector<std::pair<int,int>>& edges1, int src) {
    std::string path = "/tmp/test_" + name + ".mtx";
    write_mtx(path, n, edges1);
    GraphData g = load_graph(path);
    std::vector<std::pair<int,int>> e0;
    for (auto& [u, v] : edges1) e0.emplace_back(u-1, v-1);
    auto expected = ref_bfs(n, e0, src);

    for (auto mode : {InferenceEngine::BASELINE, InferenceEngine::STATIC,
                      InferenceEngine::DTA_MODE}) {
        BFSResult r = run_bfs(g.csr, g.csc, src, mode);
        std::set<int> got(r.visited.begin(), r.visited.end());
        got.insert(src);
        if (got != expected) {
            std::cerr << "FAIL " << name << " mode=" << (int)mode
                      << " expected=" << expected.size()
                      << " got=" << got.size() << "\n";
            assert(false);
        }
    }
    std::cout << " - " << name << ": PASSED (visited " << expected.size() << ")\n";
    std::remove(path.c_str());
}

int main() {
    std::cout << "Running BFS golden tests...\n";
    set_bfs_verbose(false);
    // Line 0-1-2-3-4.
    check("line", 5, {{1,2},{2,3},{3,4},{4,5}}, 0);
    // Star centered at 1.
    check("star", 5, {{1,2},{1,3},{1,4},{1,5}}, 0);
    // Disconnected: {1-2} + {3-4-5}; source 0 sees only 2 nodes.
    check("disconnected", 5, {{1,2},{3,4},{4,5}}, 0);
    // Self-loop at 1 (loader must not double-count/crash).
    check("selfloop", 3, {{1,1},{1,2},{2,3}}, 0);
    // Source bounds must throw (P0-7/P3-4).
    {
        GraphData g = []{
            std::string p = "/tmp/test_oob.mtx";
            write_mtx(p, 3, {{1,2},{2,3}});
            GraphData gg = load_graph(p);
            std::remove(p.c_str());
            return gg;
        }();
        bool threw = false;
        try { run_bfs(g.csr, g.csc, 99, InferenceEngine::STATIC); }
        catch (const std::out_of_range&) { threw = true; }
        assert(threw && "OOB source must throw");
        std::cout << " - source_oob: PASSED\n";
    }
    std::cout << "All BFS golden tests passed.\n";
    return 0;
}
