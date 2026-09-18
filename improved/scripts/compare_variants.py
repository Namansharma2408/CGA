"""
run_inference.py (IMPROVED) – Run all DTA variants on test graphs and collect results.

Run from CGA-2/:
    python improved/scripts/run_inference.py

Expects:
  - Binary:  CGA-2/build/Release/cga_bfs_improved
  - Graphs:  CGA-2/data/test/*.mtx
  - Output:  CGA-2/improved/results/inference_results.csv
"""

import os, sys, glob, re, subprocess

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMPROVED_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(IMPROVED_DIR)

BINARY     = os.path.join(PROJECT_ROOT, "build", "Release", "cga_bfs_improved")
TEST_GLOB  = os.path.join(PROJECT_ROOT, "data", "test", "*.mtx")
RESULTS_DIR = os.path.join(IMPROVED_DIR, "results")
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "inference_results.csv")

EXPERIMENTS = [
    ("baseline", None),
    ("cga",      None),
    ("dta",      "base"),
    ("dta",      "ucb"),
    ("dta",      "gradient"),
]


def label_for(mode, dta_type):
    if mode == "baseline": return "Baseline"
    if mode == "cga":      return "CGA-Static"
    return f"DTA-{dta_type.upper()}"


def run_inference_experiments():
    exe = BINARY
    if not os.path.exists(exe):
        exe += ".exe"
    if not os.path.exists(exe):
        print(f"ERROR: Binary not found at {BINARY}")
        print("Build first: cd improved && bash build.sh")
        sys.exit(1)

    graphs = sorted(glob.glob(TEST_GLOB))
    if not graphs:
        print(f"ERROR: No .mtx files found in {os.path.dirname(TEST_GLOB)}")
        print("Generate graphs first: python scripts/generate_graphs.py  (from CGA-2/)")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(OUTPUT_CSV, "w") as f:
        f.write("graph,variant,run,total_time_ms\n")

    print(f"Binary : {exe}")
    print(f"Graphs : {len(graphs)} in data/test/")
    print(f"Saving : {OUTPUT_CSV}\n")

    for gi, graph_path in enumerate(graphs):
        gname = os.path.basename(graph_path).replace(".mtx", "")
        print(f"[{gi+1}/{len(graphs)}] {gname}")

        for mode, dta_type in EXPERIMENTS:
            variant = label_for(mode, dta_type)
            cmd = [exe, graph_path, "0", "--mode", mode, "--runs", "3"]
            if dta_type:
                cmd += ["--dta-type", dta_type]

            try:
                res = subprocess.run(
                    cmd, check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                times = re.findall(r"Total Time:\s+([\d\.]+)\s+ms", res.stdout)
                with open(OUTPUT_CSV, "a") as f:
                    for run_i, t in enumerate(times, 1):
                        f.write(f"{gname},{variant},{run_i},{t}\n")
                print(f"    {variant:18s}  {times}")
            except subprocess.CalledProcessError as e:
                print(f"    {variant:18s}  !! FAILED: {e.stderr[:100]}")

    print(f"\n✓ Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_inference_experiments()