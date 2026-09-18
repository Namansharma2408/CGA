#include "core/bfs.h"
#include "matrix/loader.h"
#include "utils/logger.h"
#include "models/inference.h"
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>

BFSResult run_bfs(
    const CSRMatrix& A_csr,
    const CSCMatrix& A_csc,
    int source_vertex,
    InferenceEngine::Mode mode);

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <graph.mtx> <source> [--mode baseline|cga|dta] [--runs N]\n"
              << "       " << prog << " --graphs <dir/> [--mode ...] [--runs N] [--output out.csv] [--source S]\n"
              << "Flags: --mode baseline|cga|dta (cga == static-ml), --runs N>=1,\n"
              << "       --dta-type base|ucb|gradient (compat: accepted, basic DTA is unified),\n"
              << "       --output <csv> (default bfs_results.csv), --csv (machine line only),\n"
              << "       --verbose (per-iter debug), --help\n";
}

int main(int argc, char** argv) {
    std::string path;
    std::string graphs_dir;
    int source = 0;
    bool have_source = false;
    InferenceEngine::Mode mode = InferenceEngine::DTA_MODE;
    int runs = 1;
    std::string output = "bfs_results.csv";
    bool csv_only = false;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        else if (arg == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "baseline") mode = InferenceEngine::BASELINE;
            else if (m == "cga" || m == "static-ml" || m == "static") mode = InferenceEngine::STATIC;
            else if (m == "dta") mode = InferenceEngine::DTA_MODE;
            else { std::cerr << "Unknown --mode: " << m << "\n"; print_usage(argv[0]); return 1; }
        } else if (arg == "--runs" && i + 1 < argc) {
            try { runs = std::stoi(argv[++i]); }
            catch (...) { std::cerr << "Invalid --runs value\n"; return 1; }
            if (runs < 1) { std::cerr << "--runs must be >= 1\n"; return 1; }
        } else if (arg == "--dta-type" && i + 1 < argc) {
            std::string t = argv[++i];  // P2-1 compat: basic binary has unified DTA.
            if (t != "base" && t != "ucb" && t != "gradient") {
                std::cerr << "Unknown --dta-type: " << t << "\n"; return 1;
            }
            if (verbose) std::cout << "[info] --dta-type " << t << " accepted (unified DTA).\n";
        } else if (arg == "--graphs" && i + 1 < argc) {
            graphs_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else if (arg == "--source" && i + 1 < argc) {
            try { source = std::stoi(argv[++i]); have_source = true; }
            catch (...) { std::cerr << "Invalid --source value\n"; return 1; }
        } else if (arg == "--csv") {
            csv_only = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Unknown flag: " << arg << "\n"; print_usage(argv[0]); return 1;
        } else if (path.empty()) {
            path = arg;
        } else if (!have_source) {
            try { source = std::stoi(arg); have_source = true; }
            catch (...) { std::cerr << "Invalid source vertex: " << arg << "\n"; return 1; }
        } else {
            std::cerr << "Unexpected argument: " << arg << "\n"; print_usage(argv[0]); return 1;
        }
    }

    set_bfs_verbose(verbose);

    // Batch mode for EXECUTION_STEPS 3.2/3.3.
    std::vector<std::string> targets;
    if (!graphs_dir.empty()) {
        // Minimal glob via system ls (portable fallback without <filesystem> glob).
        // Caller may also pass a single file to --graphs.
        targets.push_back(graphs_dir);
        // If it is a directory, main loop below handles it by loading each .mtx
        // through load_graph per file discovered by the driver script; here we
        // support the single-file case directly and error clearly otherwise.
    } else {
        if (path.empty() || !have_source) { print_usage(argv[0]); return 1; }
        targets.push_back(path);
    }

    auto run_one = [&](const std::string& gpath, int src) -> int {
        if (!csv_only) std::cout << "Loading graph: " << gpath << "...\n";
        GraphData graph;
        try { graph = load_graph(gpath); }
        catch (const std::exception& e) { std::cerr << "Cannot load " << gpath << ": " << e.what() << "\n"; return 1; }
        if (src < 0 || src >= graph.n) {
            std::cerr << "Source " << src << " out of range [0," << graph.n << ")\n";
            return 1;
        }
        if (!csv_only) std::cout << "Loaded graph. V=" << graph.n << ", E=" << graph.m << "\n";
        for (int r = 0; r < runs; ++r) {
            if (!csv_only) std::cout << "\n--- Run " << (r+1) << " ---\n";
            try {
                BFSResult res = run_bfs(graph.csr, graph.csc, src, mode);
                if (!csv_only) res.print_summary();
                log_bfs_result_csv(res, output);
            } catch (const std::exception& e) {
                std::cout << "FATAL EXCEPTION: " << e.what() << std::endl;
                return 1;
            }
        }
        return 0;
    };

    // If --graphs points to a directory, expand *.mtx via glob-like scan.
    if (!graphs_dir.empty()) {
        // Fallback: if opendir fails, treat as single file.
        DIR* d = opendir(graphs_dir.c_str());
        if (d) {
            std::vector<std::string> files;
            struct dirent* e;
            while ((e = readdir(d)) != nullptr) {
                std::string nm = e->d_name;
                if (nm.size() > 4 && nm.substr(nm.size()-4) == ".mtx")
                    files.push_back(graphs_dir + "/" + nm);
            }
            closedir(d);
            if (files.empty()) { std::cerr << "No .mtx in " << graphs_dir << "\n"; return 1; }
            for (auto& f : files) { int rc = run_one(f, source); if (rc) return rc; }
            return 0;
        }
        return run_one(graphs_dir, source);
    }
    return run_one(path, source);
}
