#include "utils/logger.h"
#include <fstream>
#include <iostream>

void log_bfs_result_csv(const BFSResult& res, const std::string& filepath) {
    std::ofstream out(filepath);
    if (!out) {
        std::cerr << "Warning: Could not open " << filepath << " for writing.\n";
        return;
    }
    out << "Depth,FrontierNNZ,Kernel,Time_ms,Overhead_ms\n";
    for (const auto& it : res.iters) {
        out << it.depth << ","
            << it.frontier_nnz << ","
            << it.kernel << ","
            << it.ms << ","
            << it.overhead_ms << "\n";
    }
}
