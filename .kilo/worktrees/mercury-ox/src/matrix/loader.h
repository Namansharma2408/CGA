#pragma once
#include "csr.h"
#include "csc.h"
#include <string>
#include <utility>

struct GraphData {
    CSRMatrix csr;
    CSCMatrix csc;
    int n{0};
    int m{0};
};

GraphData load_graph_mtx(const std::string& path);

GraphData load_graph_edgelist(const std::string& path, int n_vertices);

GraphData load_graph(const std::string& path, int n_vertices = -1);
