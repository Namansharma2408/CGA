import os
import subprocess
import csv
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "realworld")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build", "Release")

ORIG_BIN = os.path.join(BUILD_DIR, "cga_bfs")
IMPR_BIN = os.path.join(BUILD_DIR, "cga_bfs_improved")

EXPERIMENTS = [
    ("ORIGINAL", ORIG_BIN, "baseline", None),
    ("ORIGINAL", ORIG_BIN, "cga",      None),
    ("IMPROVED", IMPR_BIN, "dta",      "ucb"),
    ("IMPROVED", IMPR_BIN, "dta",      "gradient"),
]

def run_benchmarks():
    results = []
    graphs = glob.glob(os.path.join(REAL_DATA_DIR, "**/*.mtx"), recursive=True) + \
             glob.glob(os.path.join(REAL_DATA_DIR, "*.txt"))

    if not graphs:
        print(f"No graphs found in {REAL_DATA_DIR}. Run download_realworld.py first.")
        return

    output_csv = os.path.join(SCRIPT_DIR, "realworld_results.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Framework", "Variant", "Time_ms"])

    for graph in graphs:
        gname = os.path.basename(graph)
        print(f"\n--- Benchmarking {gname} ---")

        for fw, binary, mode, dta_type in EXPERIMENTS:
            if not os.path.exists(binary) and not os.path.exists(binary + ".exe"):
                print(f"Skipping {fw}: {binary} not found.")
                continue

            cmd = [binary, graph, "0", "--mode", mode]
            if dta_type:
                cmd += ["--dta-type", dta_type]

            label = f"{fw}-{mode.upper()}"
            if dta_type: label += f"-{dta_type.upper()}"

            print(f"  Running {label}...")
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                import re
                match = re.search(r"Total Time:\s+([\d\.]+)\s+ms", res.stdout)
                if match:
                    t = float(match.group(1))
                    with open(output_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([gname, fw, label, t])
                    print(f"    -> {t} ms")
            except Exception as e:
                print(f"    !! Error running {label}: {e}")

    print(f"\n✓ Realworld benchmarking complete. Results in {output_csv}")

if __name__ == "__main__":
    run_benchmarks()