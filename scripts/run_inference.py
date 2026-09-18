import os
import sys
import glob
import subprocess
import re
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from paths import BASIC_BINARY, BASIC_BINARY_FALLBACK, TEST_GLOB, BASIC_SUMMARY_CSV

os.chdir(PROJECT_ROOT)

def _provenance_line() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    except Exception:
        gpu = "unknown"
    cuda_arch = os.environ.get("CUDA_ARCH", "86")
    return f"# provenance: date={datetime.datetime.utcnow().isoformat()}Z git={commit} gpu={gpu} cuda_arch={cuda_arch}"


def _parse_times(stdout: str):
    # P3-7: prefer machine-readable RESULT_CSV line; fallback to legacy regex.
    m = re.findall(r"RESULT_CSV:.*total_ms=([\d\.eE+-]+)", stdout)
    if m:
        return m
    return re.findall(r"Total Time:\s+([\d\.]+)\s+ms", stdout)


def run_inference_experiments():
    exe_path = BASIC_BINARY if os.path.exists(BASIC_BINARY) else BASIC_BINARY_FALLBACK

    if not os.path.exists(exe_path):
        print("Error: cga_bfs executable not found. Please compile the C++ framework first.")
        print(f"Looked for: {BASIC_BINARY} and {BASIC_BINARY_FALLBACK}")
        sys.exit(1)

    graphs = sorted(glob.glob(TEST_GLOB))
    if not graphs:
        print("No MTX graphs found in data/test/. Please run generate_graphs.py first.")
        sys.exit(1)

    print(f"Found {len(graphs)} graphs. Commencing High-Performance Inference Pipeline...")

    with open(BASIC_SUMMARY_CSV, "w") as f:
        f.write(_provenance_line() + "\n")
        f.write("Dataset,Platform,Total Time (ms)\n")

    for idx, graph_path in enumerate(graphs):
        print(f"[{idx+1}/{len(graphs)}] Executing Inference on: {os.path.basename(graph_path)}")
        source_vertex = 0
        dataset_name = os.path.basename(graph_path).replace(".mtx", "")

        for mode in ["baseline", "cga", "dta"]:
            cmd = [exe_path, graph_path, str(source_vertex), "--mode", mode, "--runs", "3"]

            try:
                result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                times = _parse_times(result.stdout)
                if times:
                    with open(BASIC_SUMMARY_CSV, "a") as f:
                        for t_str in times:
                            f.write(f"{dataset_name},{mode.upper()},{t_str}\n")
                else:
                    print(f"    !! no times parsed for {mode} (stdout tail: {result.stdout[-200:]})")

            except subprocess.CalledProcessError as e:
                print("\n!!!! CRASH IN C++ BINARY !!!!")
                print(f"Error executing {mode} for {graph_path}: {e.stderr[:500]}")

            print(f"    -> Mode: {mode.upper()} | Data Collected")

if __name__ == "__main__":
    run_inference_experiments()