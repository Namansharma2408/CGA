#include "utils/logger.h"
#include <fstream>
#include <iostream>

// P0-9/P3-9: append mode + header-once + hardware provenance hook.
// Provenance line (if present) is written by Python drivers; C++ only
// guarantees append semantics so --runs N keeps all N blocks.
void log_bfs_result_csv(const BFSResult& res, const std::string& filepath) {
    bool exists = false;
    {
        std::ifstream in(filepath);
        exists = in.good() && in.peek() != std::ifstream::traits_type::eof();
    }
    std::ofstream out(filepath, std::ios::app);
    if (!out) {
        std::cerr << "Warning: Could not open " << filepath << " for writing.\n";
        return;
    }
    if (!exists) {
        out << "Depth,FrontierNNZ,Kernel,Time_ms,Overhead_ms\n";
    }
    for (const auto& it : res.iters) {
        out << it.depth << ","
            << it.frontier_nnz << ","
            << it.kernel << ","
            << it.ms << ","
            << it.overhead_ms << "\n";
    }
    // Machine-readable summary for drivers (P3-7: no stdout scraping).
    std::cout << "RESULT_CSV: source=" << res.source
              << " visited=" << res.n_visited << "/" << res.n
              << " total_ms=" << res.total_ms << "\n";
}
