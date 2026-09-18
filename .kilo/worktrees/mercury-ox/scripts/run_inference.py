import os
import glob
import subprocess
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_inference_experiments():
    exe_path = os.path.join("build", "Release", "cga_bfs")
    if not os.path.exists(exe_path):
        exe_path += ".exe"

    if not os.path.exists(exe_path):
        exe_path = os.path.join("build", "cga_bfs")
        if not os.path.exists(exe_path):
            print(f"Error: cga_bfs executable not found. Please compile the C++ framework first.")
            return

    graphs = glob.glob("data/test/*.mtx")
    if not graphs:
        print("No MTX graphs found in data/test/. Please run generate_graphs.py first.")
        return

    print(f"Found {len(graphs)} graphs. Commencing High-Performance Inference Pipeline...")

    import re

    with open("bfs_run_summary.csv", "w") as f:
        f.write("Dataset,Platform,Total Time (ms)\n")

    for idx, graph_path in enumerate(graphs):
        print(f"[{idx+1}/{len(graphs)}] Executing Inference on: {os.path.basename(graph_path)}")
        source_vertex = 0
        dataset_name = os.path.basename(graph_path).replace(".mtx", "")

        for mode in ["baseline", "cga", "dta"]:
            cmd = [exe_path, graph_path, str(source_vertex), "--mode", mode, "--runs", "3"]

            try:
                result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                times = re.findall(r"Total Time:\s+([\d\.]+)\s+ms", result.stdout)
                if times:
                    with open("bfs_run_summary.csv", "a") as f:
                        for t_str in times:
                            f.write(f"{dataset_name},{mode.upper()},{t_str}\n")

            except subprocess.CalledProcessError as e:
                print(f"\n!!!! CRASH IN C++ BINARY !!!!")
                print(f"Error executing {mode} for {graph_path}: {e.stderr}")

            print(f"    -> Mode: {mode.upper()} | Data Collected")

if __name__ == "__main__":
    run_inference_experiments()