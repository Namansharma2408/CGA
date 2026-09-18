import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_research_plots():
    if not os.path.exists("bfs_run_summary.csv"):
        print("Error: bfs_run_summary.csv not found.")
        return

    df = pd.read_csv("bfs_run_summary.csv")
    os.makedirs("results", exist_ok=True)

    df_mean = df.groupby(['Dataset', 'Platform'])['Total Time (ms)'].mean().reset_index()

    pivot = df_mean.pivot(index='Dataset', columns='Platform', values='Total Time (ms)')
    pivot['Speedup_CGA'] = pivot['BASELINE'] / pivot['CGA']
    pivot['Speedup_DTA'] = pivot['BASELINE'] / pivot['DTA']
    pivot['DTA_vs_CGA'] = pivot['CGA'] / pivot['DTA']

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    sample_graphs = pivot.index[:min(10, len(pivot))].tolist()
    df_sample = df_mean[df_mean['Dataset'].isin(sample_graphs)]

    df_sample['Dataset_Short'] = df_sample['Dataset'].apply(lambda x: x.split('_')[2] + "\n(" + x.split('_')[3] + ")")

    plt.figure(figsize=(14, 6))
    g = sns.barplot(x='Dataset_Short', y='Total Time (ms)', hue='Platform', data=df_sample, palette=['
    plt.title("Fig. 11: BFS Total Execution Time Comparison (Log Scale)", fontsize=16, fontweight='bold')
    plt.ylabel("Total Time (ms)", fontsize=12)
    plt.xlabel("Graph Topology & Vertices", fontsize=12)
    g.set_yscale("log")
    plt.legend(title='', frameon=True)
    plt.tight_layout()
    plt.savefig("results/Fig11_BarGraph_Time.png", dpi=300)
    plt.close()

    pivot_sorted = pivot.sort_values(by='BASELINE')

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(pivot_sorted)), pivot_sorted['Speedup_CGA'], marker='o', linestyle='-', color='
    plt.plot(range(len(pivot_sorted)), pivot_sorted['Speedup_DTA'], marker='s', linestyle='-', color='

    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Baseline')

    plt.title("Fig. 12: Speedup Distribution Across Entire Dataset (Sorted by Graph Complexity)", fontsize=14, fontweight='bold')
    plt.xlabel("Tested Graphs (Sorted by Increasing Baseline Difficulty)", fontsize=12)
    plt.ylabel("Speedup Factor (x)", fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/Fig12_Speedup_LinePlot.png", dpi=300)
    plt.close()

    print("\n--- Plotting Complete ---")
    print(f"Generated results/Fig11_BarGraph_Time.png")
    print(f"Generated results/Fig12_Speedup_LinePlot.png")
    print(f"Average CGA Speedup (vs Baseline): {pivot['Speedup_CGA'].mean():.2f}x")
    print(f"Average DTA Speedup (vs Baseline): {pivot['Speedup_DTA'].mean():.2f}x")
    print(f"Average DTA vs CGA Dynamic Benefit : {pivot['DTA_vs_CGA'].mean():.2f}x")

if __name__ == "__main__":
    create_research_plots()