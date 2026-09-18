#include "core/bfs.h"
#include "matrix/loader.h"
#include "utils/logger.h"
#include "models/inference.h"
#include <iostream>
#include <string>

BFSResult run_bfs(
    const CSRMatrix& A_csr,
    const CSCMatrix& A_csc,
    int source_vertex,
    InferenceEngine::Mode mode);

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph.mtx> <source> [--mode baseline|cga|dta] [--runs N]\n";
        return 1;
    }

    std::string path = argv[1];
    int source = std::stoi(argv[2]);
    InferenceEngine::Mode mode = InferenceEngine::DTA_MODE;
    int runs = 1;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "baseline") mode = InferenceEngine::BASELINE;
            else if (m == "cga") mode = InferenceEngine::STATIC;
            else if (m == "dta") mode = InferenceEngine::DTA_MODE;
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::stoi(argv[++i]);
        }
    }

    std::cout << "Loading graph: " << path << "...\n";
    GraphData graph = load_graph(path);
    std::cout << "Loaded graph. V=" << graph.n << ", E=" << graph.m << "\n";

    for (int r = 0; r < runs; ++r) {
        std::cout << "\n--- Run " << (r+1) << " ---\n";
        try {
            BFSResult res = run_bfs(graph.csr, graph.csc, source, mode);
            res.print_summary();
            log_bfs_result_csv(res, "bfs_results.csv");
        } catch (const std::exception& e) {
            std::cout << "FATAL EXCEPTION: " << e.what() << std::endl;
            return 1;
        }
    }

    return 0;
}
