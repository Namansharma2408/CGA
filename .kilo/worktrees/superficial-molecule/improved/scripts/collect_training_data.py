"""
improved/scripts/collect_training_data.py

Ground-truth training data collector.
For each training graph, runs ALL CPU kernels in random order via the compiled
cga_bfs binary (in profiling mode) and records per-depth timing.
The result is data/train_labeled.csv which can be fed to train_models.py
instead of the simulated timing.

Requires:
    - cga_bfs binary compiled with --mode profile
    - data/train/*.mtx graphs
"""

import os
import sys
import glob
import subprocess
import random
import csv
import argparse
import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, SCRIPT_DIR)

from feature_extraction import FEATURE_NAMES, DIM

CPU_KERNELS = ["PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA", "Gustavson"]
GPU_KERNELS = ["Sort-Based", "CSR-Vector", "Merge-Based"]
ALL_KERNELS = CPU_KERNELS + GPU_KERNELS


def find_binary():
    candidates = [
        os.path.join(PROJECT_ROOT, "build", "Release", "cga_bfs.exe"),
        os.path.join(PROJECT_ROOT, "build", "cga_bfs.exe"),
        os.path.join(PROJECT_ROOT, "build", "cga_bfs"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def run_kernel_timing(binary: str, graph: str, source: int,
                      kernel: str) -> list:
    """
    Run cga_bfs with a forced specific kernel and parse timing output.
    Returns list of (depth, frontier_nnz, time_ms) tuples.
    """
    cmd = [binary, graph, str(source),
           "--mode", "baseline",
           "--runs", "1"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return []
        rows = []
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 4 and parts[0].strip().isdigit():
                try:
                    depth     = int(parts[0].strip())
                    fnnz      = int(parts[1].strip())
                    time_ms   = float(parts[3].strip())
                    rows.append((depth, fnnz, time_ms))
                except ValueError:
                    pass
        return rows
    except Exception:
        return []


def collect_all(binary: str, graphs: list, output_csv: str):
    """
    For each graph, run all kernels in random order and write labeled CSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    fieldnames = (["graph", "depth", "frontier_nnz"]
                  + FEATURE_NAMES
                  + [f"t_{k}" for k in ALL_KERNELS]
                  + ["y1_platform", "y2_cpu", "y3_gpu", "y4_universal"])

    written = 0
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for gi, graph in enumerate(graphs):
            print(f"[{gi+1}/{len(graphs)}] {os.path.basename(graph)}")
            kern_order = ALL_KERNELS.copy()
            random.shuffle(kern_order)

            timing_by_depth = {}

            for kernel in kern_order:
                rows = run_kernel_timing(binary, graph, 0, kernel)
                for depth, fnnz, t_ms in rows:
                    if depth not in timing_by_depth:
                        timing_by_depth[depth] = {"fnnz": fnnz}
                    timing_by_depth[depth][kernel] = t_ms

            if not timing_by_depth:
                print(f"  [warn] no timing data — skipping")
                continue

            for depth, data in sorted(timing_by_depth.items()):
                t_arr_cpu = [data.get(k, np.inf) for k in CPU_KERNELS]
                t_arr_gpu = [data.get(k, np.inf) for k in GPU_KERNELS]
                t_all     = t_arr_cpu + t_arr_gpu

                y1 = 1 if min(t_arr_gpu) < min(t_arr_cpu) else 0
                y2 = int(np.argmin(t_arr_cpu))
                y3 = int(np.argmin(t_arr_gpu))
                y4 = 1 if t_arr_gpu[1] < t_arr_cpu[4] else 0

                row = {
                    "graph":        os.path.basename(graph),
                    "depth":        depth,
                    "frontier_nnz": data.get("fnnz", 0),
                    "y1_platform":  y1,
                    "y2_cpu":       y2,
                    "y3_gpu":       y3,
                    "y4_universal": y4,
                }
                for fn in FEATURE_NAMES:
                    row[fn] = 0.0
                for i, k in enumerate(ALL_KERNELS):
                    row[f"t_{k}"] = t_all[i]

                writer.writerow(row)
                written += 1

    print(f"\n✓ Wrote {written} labeled rows to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect ground-truth kernel timing labels from CGA-BFS binary.")
    parser.add_argument("--output", default="data/train_labeled.csv")
    parser.add_argument("--graphs", default="data/train/*.mtx")
    args = parser.parse_args()

    binary = find_binary()
    if not binary:
        print("Error: cga_bfs binary not found. Build the project first.")
        print("  cmake --build build --target cga_bfs")
        sys.exit(1)
    print(f"Using binary: {binary}")

    graphs = glob.glob(os.path.join(PROJECT_ROOT, args.graphs))
    if not graphs:
        print(f"No graphs found at: {args.graphs}"); sys.exit(1)

    collect_all(binary, graphs, args.output)


if __name__ == "__main__":
    main()