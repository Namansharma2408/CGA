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
    InferenceEngine::Mode mode,
    InferenceEngine::DTA_TYPE dta_type = InferenceEngine::BASE_DTA);

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <graph.mtx> <source> [--mode baseline|cga|dta] [--runs N]\n"
              << "       " << prog << " --graphs <dir/> [--mode ...] [--runs N] [--output out.csv]\n"
              << "Flags: --mode baseline|cga|dta, --dta-type base|ucb|gradient,\n"
              << "       --runs N>=1, --output <csv>, --csv, --verbose, --help\n";
}

int main(int argc, char** argv) {
    std::string path, graphs_dir, output = "bfs_results.csv";
    int source = 0;
    bool have_source = false;
    InferenceEngine::Mode mode = InferenceEngine::DTA_MODE;
    InferenceEngine::DTA_TYPE dta_type = InferenceEngine::BASE_DTA;
    int runs = 1;
    bool csv_only = false, verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        else if (arg == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "baseline") mode = InferenceEngine::BASELINE;
            else if (m == "cga" || m == "static-ml" || m == "static") mode = InferenceEngine::STATIC;
            else if (m == "dta") mode = InferenceEngine::DTA_MODE;
            else { std::cerr << "Unknown --mode: " << m << "\n"; return 1; }
        } else if (arg == "--dta-type" && i + 1 < argc) {
            std::string t = argv[++i];
            if (t == "base") dta_type = InferenceEngine::BASE_DTA;
            else if (t == "ucb") dta_type = InferenceEngine::UCB_DTA;
            else if (t == "gradient") dta_type = InferenceEngine::GRADIENT_DTA;
            else { std::cerr << "Unknown --dta-type: " << t << "\n"; return 1; }
        } else if (arg == "--runs" && i + 1 < argc) {
            try { runs = std::stoi(argv[++i]); } catch (...) { std::cerr << "Invalid --runs\n"; return 1; }
            if (runs < 1) { std::cerr << "--runs >= 1\n"; return 1; }
        } else if (arg == "--graphs" && i + 1 < argc) { graphs_dir = argv[++i]; }
        else if (arg == "--output" && i + 1 < argc) { output = argv[++i]; }
        else if (arg == "--source" && i + 1 < argc) {
            try { source = std::stoi(argv[++i]); have_source = true; }
            catch (...) { std::cerr << "Invalid --source\n"; return 1; }
        } else if (arg == "--csv") { csv_only = true; }
        else if (arg == "--verbose") { verbose = true; set_bfs_verbose(true); }
        else if (arg.rfind("--", 0) == 0) { std::cerr << "Unknown flag: " << arg << "\n"; print_usage(argv[0]); return 1; }
        else if (path.empty()) { path = arg; }
        else if (!have_source) {
            try { source = std::stoi(arg); have_source = true; }
            catch (...) { std::cerr << "Invalid source: " << arg << "\n"; return 1; }
        } else { std::cerr << "Unexpected: " << arg << "\n"; print_usage(argv[0]); return 1; }
    }

    auto run_one = [&](const std::string& gpath, int src) -> int {
        if (!csv_only) std::cout << "Loading graph: " << gpath << "...\n";
        GraphData g;
        try { g = load_graph(gpath); } catch (const std::exception& e) {
            std::cerr << "Cannot load " << gpath << ": " << e.what() << "\n"; return 1;
        }
        if (src < 0 || src >= g.n) { std::cerr << "Source OOB\n"; return 1; }
        if (!csv_only) std::cout << "Loaded V=" << g.n << " E=" << g.m << "\n";
        for (int r = 0; r < runs; ++r) {
            if (!csv_only) std::cout << "\n--- Run " << (r+1) << " ---\n";
            try {
                BFSResult res = run_bfs(g.csr, g.csc, src, mode, dta_type);
                if (!csv_only) res.print_summary();
                log_bfs_result_csv(res, output);
            } catch (const std::exception& e) {
                std::cout << "FATAL: " << e.what() << "\n"; return 1;
            }
        }
        return 0;
    };

    if (!graphs_dir.empty()) {
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
    if (path.empty() || !have_source) { print_usage(argv[0]); return 1; }
    (void)verbose;
    return run_one(path, source);
}
