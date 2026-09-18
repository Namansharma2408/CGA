#include "matrix/loader.h"
#include "cpu/spmspv/kernels.h"
#include "utils/timer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

std::vector<bool> generate_mask(int n, double density) {
    std::vector<bool> mask(n, false);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        if (dist(rng) < density) mask[i] = true;
    }
    return mask;
}

SparseVector generate_x(int n, double density) {
    std::vector<int> indices;
    std::vector<float> values;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        if (dist(rng) < density) {
            indices.push_back(i);
            values.push_back(1.0f);
        }
    }
    return SparseVector(n, std::move(indices), std::move(values));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph.mtx> [output.csv]\n";
        return 1;
    }

    std::string path = argv[1];
    std::string out_path = (argc > 1) ? argv[2] : "cpu_mflops_benchmark.csv";

    std::cout << "Loading graph: " << path << " for CPU Isolated Micro-Benchmarks...\n";
    GraphData graph = load_graph(path);
    const CSCMatrix& A = graph.csc;

    std::vector<double> m_densities = {0.001, 0.01, 0.1, 1.0};
    std::vector<double> x_densities = {0.004, 0.008, 0.016, 0.032, 0.064, 0.128};
    std::vector<std::string> kernel_names = {
        "PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA", "LB-MSPA"
    };

    std::ofstream out(out_path);
    out << "MaskDensity,XDensity,Kernel,Time_ms,MFlops\n";

    int runs = 5;

    for (double md : m_densities) {
        std::vector<bool> mask = generate_mask(A.n_cols, md);

        for (double xd : x_densities) {
            SparseVector x = generate_x(A.n_cols, xd);
            if (x.nnz == 0) continue;

            long long operations = 0;
            for (int i = 0; i < x.nnz; ++i) {
                int col = x.indices[i];
                operations += (A.indptr[col + 1] - A.indptr[col]) * 2;
            }
            if (operations == 0) continue;

            for (const auto& k_name : kernel_names) {
                double total_ms = 0;
                for (int r = 0; r < runs; ++r) {
                    Timer t;
                    KernelResult res;
                    t.start();
                    if (k_name == "PM-BHash") {
                        res = cpu_pm_bhash(A, x, mask);
                    } else if (k_name == "LB-PM-BHash") {
                        res = cpu_lb_pm_bhash(A, x, mask, 16);
                    } else if (k_name == "PB-MSPA") {
                        res = cpu_pb_mspa(A, x, mask);
                    } else if (k_name == "LB-PB-MSPA") {
                        res = cpu_lb_pb_mspa(A, x, mask, 16);
                    } else if (k_name == "LB-MSPA") {
                        res = cpu_gustavson(A, x, mask);
                    }
                    double ms = t.stopMs();
                    if (r >= 2) total_ms += ms;
                }

                double avg_ms = total_ms / (runs - 2);
                double mflops = (operations / 1e6) / (avg_ms / 1000.0);

                out << md << "," << xd << "," << k_name << "," << avg_ms << "," << mflops << "\n";
            }
        }
        std::cout << "Finished Mask Density: " << md << "\n";
    }

    std::cout << "Done! Results saved to cpu_mflops_benchmark.csv\n";
    return 0;
}
