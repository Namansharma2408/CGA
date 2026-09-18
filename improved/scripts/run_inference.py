"""
run_inference.py (IMPROVED) – Run all DTA variants on test graphs and collect results.

Run from repo root (either works; paths resolve from __file__):
    python improved/scripts/run_inference.py

Expects:
  - Binary:  build/Release/cga_bfs_improved (canonical, see P2-2/SPEC.md)
  - Graphs:  data/test/*.mtx
  - Output:  improved/results/inference_results.csv
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


def _parse_times(stdout):
    # P3-7: prefer RESULT_CSV machine line; fallback to legacy Total Time regex.
    m = re.findall(r"RESULT_CSV:.*total_ms=([\d\.eE+-]+)", stdout)
    if m:
        return m
    return re.findall(r"Total Time:\s+([\d\.]+)\s+ms", stdout)


def run_inference_experiments():
    exe = BINARY
    # P2-8: only try .exe on Windows; on Linux fail loudly with build hint.
    if not os.path.exists(exe):
        if os.name == "nt" and os.path.exists(exe + ".exe"):
            exe += ".exe"
    if not os.path.exists(exe):
        print(f"ERROR: Binary not found at {BINARY}")
        print("Build first: bash improved/build.sh  (from repo root)")
        print("  or: cd improved && bash build.sh")
        sys.exit(1)

    graphs = sorted(glob.glob(TEST_GLOB))
    if not graphs:
        print(f"ERROR: No .mtx files found in {os.path.dirname(TEST_GLOB)}")
        print("Generate graphs first: python scripts/generate_graphs.py  (from CGA-2/)")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    import datetime, subprocess as _sp
    try:
        _commit = _sp.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        _commit = "unknown"
    with open(OUTPUT_CSV, "w") as f:
        f.write(f"# provenance: date={datetime.datetime.utcnow().isoformat()}Z git={_commit} "
                f"cuda_arch={os.environ.get('CUDA_ARCH','86')}\n")
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
                times = _parse_times(res.stdout)
                with open(OUTPUT_CSV, "a") as f:
                    for run_i, t in enumerate(times, 1):
                        f.write(f"{gname},{variant},{run_i},{t}\n")
                print(f"    {variant:18s}  {times}")
            except subprocess.CalledProcessError as e:
                print(f"    {variant:18s}  !! FAILED: {e.stderr[:100]}")

    print(f"\n✓ Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_inference_experiments()