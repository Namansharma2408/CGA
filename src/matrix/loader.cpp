#include "matrix/loader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <set>
#include <map>

// P0-7: symmetrizes directed edges for undirected BFS (adds reverse edge,
// dedups, sorts). Self-loops kept once (no reverse duplicate). 1-indexed MTX
// inputs are converted to 0-indexed before this call; edgelist inputs are
// assumed 0-indexed. Non-square MTX warns but proceeds using row count.
static GraphData build_from_edges(
    int n, std::vector<std::pair<int,int>>& edges)
{
    size_t orig = edges.size();
    for (size_t i = 0; i < orig; ++i)
        if (edges[i].first != edges[i].second)
            edges.push_back({edges[i].second, edges[i].first});

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    int nnz = (int)edges.size();
    CSRMatrix csr(n, n, nnz);
    csr.indptr[0] = 0;
    int e = 0, cur_row = 0;
    for (auto& [u, v] : edges) {
        while (cur_row < u) csr.indptr[++cur_row] = e;
        csr.indices[e] = v;
        csr.data[e]    = 1.0f;
        ++e;
    }
    while (cur_row < n) csr.indptr[++cur_row] = nnz;

    GraphData g;
    g.n   = n;
    g.m   = nnz;
    g.csr = csr;
    g.csc = CSCMatrix::from_csr(csr);
    return g;
}

GraphData load_graph_mtx(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    std::string line;
    int n = 0, m_file = 0, nnz_declared = 0;
    bool header_done = false;
    std::vector<std::pair<int,int>> edges;

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '%') continue;

        std::istringstream ss(line);
        if (!header_done) {
            int tmp;
            ss >> n >> tmp >> nnz_declared;
            m_file = tmp;
            if (n != m_file) {
                std::cerr << "[loader] Warning: non-square MTX ("
                          << n << "x" << m_file << "), using rows as n.\n";
            }
            header_done = true;
            edges.reserve(nnz_declared * 2);
            continue;
        }
        int u, v; float w = 1.0f;
        ss >> u >> v;
        if (ss) ss >> w;
        u--; v--;
        if (u >= 0 && v >= 0 && u < n && v < n)
            edges.push_back({u, v});
    }

    std::cout << "[loader] MTX: " << path << "  n=" << n
              << " declared_nnz=" << nnz_declared
              << " read_edges=" << edges.size() << "\n";
    return build_from_edges(n, edges);
}

GraphData load_graph_edgelist(const std::string& path, int n_vertices) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    std::vector<std::pair<int,int>> edges;
    int u, v, max_node = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        if (ss >> u >> v) {
            edges.push_back({u, v});
            max_node = std::max(max_node, std::max(u, v));
        }
    }
    int n = (n_vertices > 0) ? n_vertices : max_node + 1;
    std::cout << "[loader] EdgeList: " << path
              << "  n=" << n << " edges=" << edges.size() << "\n";
    return build_from_edges(n, edges);
}

GraphData load_graph(const std::string& path, int n_vertices) {
    if (path.size() >= 4 && path.substr(path.size()-4) == ".mtx")
        return load_graph_mtx(path);
    return load_graph_edgelist(path, n_vertices);
}
