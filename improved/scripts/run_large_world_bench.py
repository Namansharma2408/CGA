import os, sys, glob, re, subprocess

# Target configuration for Large World Benchmarking
BINARY     = "build/Release/cga_bfs_improved"
TEST_GLOB  = "data/large_world/*.mtx"
OUTPUT_CSV = "improved/results/large_world_inference_results.csv"

# Same experimental setup as standard inference
EXPERIMENTS = [
    ("baseline", None),
    ("cga",      None),
    ("dta",      "ucb"),
]

def label_for(mode, dta_type):
    if mode == "baseline": return "Baseline"
    if mode == "cga":      return "CGA-Static"
    return f"DTA-{dta_type.upper()}"

def run_large_bench():
    exe = BINARY
    if os.name == 'nt' and not exe.endswith(".exe"): exe += ".exe"
    
    if not os.path.exists(exe):
        print(f"ERROR: Binary not found at {exe}. Please build the project first.")
        sys.exit(1)

    graphs = sorted(glob.glob(TEST_GLOB))
    if not graphs:
        print(f"ERROR: No large graphs found in {os.path.dirname(TEST_GLOB)}")
        print("Run python scripts/generate_large_world.py first.")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w") as f:
        f.write("graph,variant,run,total_time_ms\n")

    print(f"--- Large World Benchmark ---")
    print(f"Binary : {exe}")
    print(f"Graphs : {len(graphs)} found.")
    print(f"Output : {OUTPUT_CSV}\n")

    for gi, graph_path in enumerate(graphs):
        gname = os.path.basename(graph_path).replace(".mtx", "")
        print(f"[{gi+1}/{len(graphs)}] Running {gname}...")

        for mode, dta_type in EXPERIMENTS:
            variant = label_for(mode, dta_type)
            cmd = [exe, graph_path, "0", "--mode", mode, "--runs", "3"]
            if dta_type: cmd += ["--dta-type", dta_type]

            try:
                res = subprocess.run(cmd, check=True, capture_output=True, text=True)
                times = re.findall(r"Total Time:\s+([\d\.]+)\s+ms", res.stdout)
                with open(OUTPUT_CSV, "a") as f:
                    for run_i, t in enumerate(times, 1):
                        f.write(f"{gname},{variant},{run_i},{t}\n")
                print(f"    {variant:18s}  Avg: {sum(map(float, times))/len(times):.2f} ms")
            except Exception as e:
                print(f"    {variant:18s}  !! FAILED")

    print(f"\n✓ Completed. Results in {OUTPUT_CSV}")

if __name__ == "__main__":
    run_large_bench()
