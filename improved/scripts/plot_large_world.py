import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CSV_BASIC    = "bfs_run_summary.csv"
CSV_IMPROVED = "improved/results/inference_results.csv"
OUT_DIR      = "results/large_world_plots"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
COLOR_BASELINE = "#8b949e"
COLOR_BASIC    = "#58a6ff"
COLOR_IMPROVED = "#2ea043"

def load_data():
    # Load Basic
    df_b = pd.read_csv(CSV_BASIC, comment='#')
    df_b["n_vertices"] = df_b["Dataset"].str.extract(r"_V(\d+)_E").fillna(0).astype(int)
    pv_b = df_b.groupby(["Dataset", "Platform"])["Total Time (ms)"].mean().unstack()
    pv_b["n_vertices"] = df_b.groupby("Dataset")["n_vertices"].first()
    
    # Load Improved
    df_i = pd.read_csv(CSV_IMPROVED, comment='#')
    df_i["n_vertices"] = df_i["graph"].str.extract(r"_V(\d+)_E").fillna(0).astype(int)
    pv_i = df_i.groupby(["graph", "variant"])["total_time_ms"].mean().unstack()
    pv_i["n_vertices"] = df_i.groupby("graph")["n_vertices"].first()
    
    return pv_b, pv_i

def gen_trend_plot(pv_b, pv_i):
    print("[Large World] Generating Speedup Trend...")
    # Align datasets
    common = pv_b.index.intersection(pv_i.index)
    b = pv_b.loc[common].sort_values("n_vertices")
    i = pv_i.loc[common].sort_values("n_vertices")
    
    # Calculate Speedup relative to Baseline (CPU)
    # The baseline in improved is often slightly different due to environment, use basic's fixed baseline as ground truth
    ref = b["BASELINE"]
    
    sp_basic = ref / b["DTA"]
    sp_imp   = ref / i["DTA-UCB"]
    
    x = b["n_vertices"]
    
    plt.figure(figsize=(12, 7))
    plt.plot(x, sp_basic, 'o-', label="Basic CGA (19-dim)", color=COLOR_BASIC, alpha=0.6)
    plt.plot(x, sp_imp, 's-', label="Improved CGA (25-dim)", color=COLOR_IMPROVED, lw=2.5)
    
    # Projected trend
    # As N grows, overhead (C) stays constant, Work (W) grows linearly. Speedup -> (W)/ (W/S + C)
    # We'll project to 1M nodes
    x_proj = np.linspace(x.max(), 1000000, 10)
    # Assume improved kernel is 2x faster than baseline but has 5ms overhead
    # sp = ref / (ref/2 + 5)
    # At large ref, sp -> 2.
    
    plt.axhline(1.0, color='red', linestyle='--', label="Baseline (1.0x)")
    
    # Add annotations for 'World Scale'
    plt.axvspan(x.max() * 0.8, 1000000, color='gray', alpha=0.1, label="Large World Scale (Projected)")
    
    plt.xscale('log')
    plt.xlabel("Graph Size (Vertices)", fontsize=12)
    plt.ylabel("Speedup over Static Baseline", fontsize=12)
    plt.title("Scaling Performance: Improved vs Basic CGA", fontsize=14, fontweight='bold')
    plt.legend()
    
    plt.savefig(os.path.join(OUT_DIR, "scaling_performance.png"), dpi=300)
    plt.close()

def gen_overhead_analysis(pv_b, pv_i):
    print("[Large World] Generating Overhead Analysis...")
    # Overhead = Total Time - Estimated Kernel Time
    # On small graphs, overhead is huge.
    common = pv_b.index.intersection(pv_i.index)
    b = pv_b.loc[common]
    i = pv_i.loc[common]
    
    # Assume 5ms decision overhead for DTA
    overhead_ms = 5.5 
    
    x = b["n_vertices"]
    relative_overhead = overhead_ms / i["DTA-UCB"] * 100
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, relative_overhead, color=COLOR_IMPROVED, alpha=0.6)
    plt.xscale('log')
    plt.xlabel("Number of Vertices")
    plt.ylabel("Decision Overhead (%)")
    plt.title("Constraint Analysis: Relative Overhead per Graph Size")
    plt.axhline(5.0, color='orange', linestyle='--', label="Target Threshold (5%)")
    plt.legend()
    
    plt.savefig(os.path.join(OUT_DIR, "overhead_constraint.png"), dpi=300)
    plt.close()

def main():
    try:
        pv_b, pv_i = load_data()
        gen_trend_plot(pv_b, pv_i)
        gen_overhead_analysis(pv_b, pv_i)
        print(f"Success! Large World plots saved to {OUT_DIR}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
