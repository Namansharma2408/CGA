"""
plot_paper_figures.py
=====================
CGA-2 Paper Plot Script — reads from bfs_run_summary.csv in CGA-2 root.
Generates ONLY the paper plots for the BASIC CGA framework.

All improved-version plots live in:  improved/scripts/plot_improved.py

Run from CGA-2/ directory:
    python scripts/plot_paper_figures.py

Outputs -> results/paper_plots/
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH     = os.path.join(PROJECT_ROOT, "bfs_run_summary.csv")
OUT_DIR      = os.path.join(PROJECT_ROOT, "results", "paper_plots")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models_trained")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print("╔══════════════════════════════════════════════════╗")
    print("║  ERROR: bfs_run_summary.csv not found!           ║")
    print("║  Run inference first:                            ║")
    print("║    python scripts/run_inference.py               ║")
    print("╚══════════════════════════════════════════════════╝")
    sys.exit(1)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
PAL = {"BASELINE": "#8b949e", "CGA": "#58a6ff", "DTA": "#d2a8ff"}


def save(fig, name):
    p = os.path.join(OUT_DIR, name + ".png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["topology"]   = df["Dataset"].str.extract(r"graph_\d+_([a-z_]+)_V\d+")
    df["n_vertices"] = df["Dataset"].str.extract(r"_V(\d+)_E").astype(int)
    df["n_edges"]    = df["Dataset"].str.extract(r"_E(\d+)$").astype(int)
    df["size_bin"]   = pd.cut(df["n_vertices"],
                               bins=[0,7000,12000,16000,25000],
                               labels=["XS(<7k)","S(7-12k)","M(12-16k)","L(>16k)"])
    return df


def compute_pivot(df):
    mean_df = df.groupby(["Dataset","Platform"])["Total Time (ms)"].mean().reset_index()
    pv = mean_df.pivot(index="Dataset", columns="Platform", values="Total Time (ms)")
    for c in ["BASELINE","CGA","DTA"]:
        if c not in pv.columns: pv[c] = np.nan
    pv["Speedup_CGA"] = pv["BASELINE"] / pv["CGA"]
    pv["Speedup_DTA"] = pv["BASELINE"] / pv["DTA"]
    pv["DTA_vs_CGA"]  = pv["CGA"] / pv["DTA"]
    meta = df[["Dataset","topology","n_vertices","n_edges","size_bin"]].drop_duplicates("Dataset").set_index("Dataset")
    return pv.join(meta)


def gen_fig11(df):
    print("[Fig 11] Execution Time Bar...")
    mean_df = df.groupby(["Dataset","Platform"])["Total Time (ms)"].mean().reset_index()
    sample  = mean_df["Dataset"].unique()[:12]
    sub = mean_df[mean_df["Dataset"].isin(sample)].copy()
    sub["Label"] = sub["Dataset"].str.extract(r"graph_\d+_([a-z_]+)_V(\d+)").apply(
        lambda r: f"{r[0][:3].upper()}\n({int(r[1])//1000}k)", axis=1)
    fig, ax = plt.subplots(figsize=(14,6))
    sns.barplot(x="Label", y="Total Time (ms)", hue="Platform",
                data=sub, palette=list(PAL.values()), ax=ax, errorbar="sd")
    ax.set_yscale("log")
    ax.set_title("Fig. 11: BFS Total Execution Time (Log Scale)", fontweight="bold")
    ax.set_xlabel("Graph"); ax.set_ylabel("Total Time (ms)")
    ax.legend(title="")
    save(fig, "Fig11_BarGraph_Time")


def gen_fig12(pv):
    print("[Fig 12] Speedup Distribution...")
    pv_s = pv.sort_values("BASELINE")
    x    = range(len(pv_s))
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(x, pv_s["Speedup_CGA"], "o-", color=PAL["CGA"],  lw=1.5, ms=3, label="CGA vs Baseline")
    ax.plot(x, pv_s["Speedup_DTA"], "s-", color=PAL["DTA"],  lw=1.5, ms=3, label="DTA vs Baseline")
    ax.axhline(1.0, color="red", ls="--", lw=1.2, label="1×")
    ax.set_yscale("log")
    ax.set_xlabel("Graphs sorted by baseline difficulty")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Fig. 12: Speedup Distribution Across Dataset", fontweight="bold")
    ax.legend()
    save(fig, "Fig12_Speedup_LinePlot")


def gen_topology_box(pv):
    print("[FigA] Speedup by Topology...")
    melted = pd.melt(pv.reset_index()[["topology","Speedup_CGA","Speedup_DTA"]],
                     id_vars="topology", var_name="Method", value_name="Speedup")
    melted["Method"] = melted["Method"].map({"Speedup_CGA":"CGA","Speedup_DTA":"DTA"})
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(x="topology", y="Speedup", hue="Method", data=melted,
                palette={"CGA":PAL["CGA"],"DTA":PAL["DTA"]}, ax=ax)
    ax.axhline(1.0, color="red", ls="--", lw=1.2)
    ax.set_title("Fig. A: Speedup by Graph Topology", fontweight="bold")
    save(fig, "FigA_Speedup_Topology_Boxplot")


def gen_heatmap(pv):
    print("[FigB] Speedup Heatmap...")
    hm_cga = pv.groupby(["topology","size_bin"])["Speedup_CGA"].mean().unstack(fill_value=np.nan)
    hm_dta = pv.groupby(["topology","size_bin"])["Speedup_DTA"].mean().unstack(fill_value=np.nan)
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    for ax, hm, title in zip(axes,[hm_cga,hm_dta],["CGA Speedup","DTA Speedup"]):
        sns.heatmap(hm, annot=True, fmt=".2f", cmap="YlGn", ax=ax, linewidths=0.5)
        ax.set_title(f"{title} (Topology × Size)", fontweight="bold")
    save(fig, "FigB_Speedup_Heatmap")


def gen_scatter_dta_cga(pv):
    print("[FigC] DTA vs CGA Scatter...")
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(pv["Speedup_CGA"], pv["Speedup_DTA"],
                    c=pv["n_vertices"], cmap="viridis", alpha=0.75, s=50, edgecolors="k", lw=0.4)
    ax.axline((1,1), slope=1, color="red", ls="--", lw=1.2, label="DTA = CGA")
    plt.colorbar(sc, ax=ax, label="Number of Vertices")
    ax.set_xlabel("CGA Speedup (×)"); ax.set_ylabel("DTA Speedup (×)")
    ax.set_title("Fig. C: DTA vs CGA Speedup", fontweight="bold")
    ax.legend()
    save(fig, "FigC_DTA_vs_CGA_Scatter")


def gen_sorted_bar(pv):
    print("[FigD] Per-graph sorted bar...")
    pv_s   = pv.sort_values("Speedup_DTA", ascending=False)
    labels = [f"{t[:3].upper()}-{i}" for i, t in enumerate(pv_s["topology"])]
    x = np.arange(len(pv_s))
    fig, ax = plt.subplots(figsize=(16,5))
    ax.bar(x-0.2, pv_s["Speedup_CGA"], 0.35, color=PAL["CGA"], label="CGA", alpha=0.85)
    ax.bar(x+0.2, pv_s["Speedup_DTA"], 0.35, color=PAL["DTA"], label="DTA", alpha=0.85)
    ax.axhline(1.0, color="red", ls="--", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Speedup (×)"); ax.set_title("Fig. D: Per-Graph Speedup", fontweight="bold")
    ax.legend()
    save(fig, "FigD_PerGraph_Speedup_Bar")


def gen_run_consistency(df):
    print("[FigE] Run consistency CoV...")
    std_df  = df.groupby(["Dataset","Platform"])["Total Time (ms)"].std().reset_index()
    mean_df = df.groupby(["Dataset","Platform"])["Total Time (ms)"].mean().reset_index()
    comb    = std_df.merge(mean_df, on=["Dataset","Platform"], suffixes=("_std","_mean"))
    comb["CV"] = comb["Total Time (ms)_std"] / comb["Total Time (ms)_mean"]
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(x="Platform", y="CV", data=comb, palette=PAL, ax=ax,
                order=["BASELINE","CGA","DTA"])
    ax.set_title("Fig. E: Coefficient of Variation (lower = more stable)", fontweight="bold")
    save(fig, "FigE_Run_Consistency_CoV")


def gen_topology_grouped(pv):
    print("[FigF] Topology grouped bar...")
    g = pv.groupby("topology")[["Speedup_CGA","Speedup_DTA"]].mean()
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x-0.2, g["Speedup_CGA"], 0.35, color=PAL["CGA"], label="CGA", alpha=0.85)
    ax.bar(x+0.2, g["Speedup_DTA"], 0.35, color=PAL["DTA"], label="DTA", alpha=0.85)
    ax.axhline(1.0, color="red", ls="--")
    ax.set_xticks(x); ax.set_xticklabels(g.index)
    for i,(a,b) in enumerate(zip(g["Speedup_CGA"],g["Speedup_DTA"])):
        ax.text(i-0.2, a+0.01, f"{a:.2f}×", ha="center", fontsize=8)
        ax.text(i+0.2, b+0.01, f"{b:.2f}×", ha="center", fontsize=8)
    ax.set_ylabel("Avg Speedup (×)"); ax.set_title("Fig. F: Avg Speedup by Topology", fontweight="bold")
    ax.legend()
    save(fig, "FigF_Topology_Speedup_Group")


def gen_frontier_profile():
    print("[FigL] BFS Frontier Profile (simulated)...")
    depths = np.arange(0,15)
    topologies = {
        "Scale-Free":  lambda d: int(5000*4*(d/14)*(1-d/14)),
        "Erdos-Renyi": lambda d: int(3000*4*(d/14)*(1-d/14)),
        "Small-World": lambda d: int(2000*4*(d/14)*(1-d/14)),
    }
    colors = ["#58a6ff", "#2ea043", "#d2a8ff"]
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,7),sharex=True)
    for (topo,fn),col in zip(topologies.items(),colors):
        sizes = [max(1,fn(d)) for d in depths]
        ax1.plot(depths,sizes,"o-",color=col,lw=2,ms=5,label=topo)
        ax2.plot(depths,[s/20000 for s in sizes],"s--",color=col,lw=2,ms=5,label=topo)
    ax1.set_ylabel("Frontier NNZ"); ax1.set_title("Fig. L: BFS Frontier Growth", fontweight="bold")
    ax1.legend()
    ax2.set_xlabel("BFS Depth"); ax2.set_ylabel("x_density")
    ax2.axhline(0.05,color="gray",ls="--",lw=1.2,label="τ≈0.05"); ax2.legend()
    plt.tight_layout()
    save(fig, "FigL_BFS_Frontier_Profile")


def gen_eda(df):
    print("[FigM] Dataset EDA...")
    meta = df[["Dataset","topology","n_vertices","n_edges"]].drop_duplicates("Dataset")
    fig  = plt.figure(figsize=(14,9))
    gs   = GridSpec(2,3,figure=fig)

    ax1  = fig.add_subplot(gs[0,0])
    meta["n_vertices"].hist(bins=15,color="#58a6ff",edgecolor="k",ax=ax1)
    ax1.set_title("N Vertices")

    ax2 = fig.add_subplot(gs[0,1])
    meta["n_edges"].hist(bins=15,color="#2ea043",edgecolor="k",ax=ax2)
    ax2.set_title("N Edges")

    ax3 = fig.add_subplot(gs[0,2])
    meta["topology"].value_counts().plot.bar(ax=ax3,edgecolor="k",color=["#58a6ff", "#2ea043", "#d2a8ff"])
    ax3.set_title("Topology"); ax3.tick_params(axis="x",rotation=15)

    ax4 = fig.add_subplot(gs[1,0])
    (meta["n_edges"]/meta["n_vertices"]).hist(bins=15,color="#d2a8ff",edgecolor="k",ax=ax4)
    ax4.set_title("Edge Density (E/V)"); ax4.set_xlabel("E/V")

    ax5 = fig.add_subplot(gs[1,1:])
    sns.scatterplot(data=meta,x="n_vertices",y="n_edges",hue="topology",palette="Set1",ax=ax5,s=60)
    ax5.set_title("V vs E by Topology")

    fig.suptitle("Fig. M: Dataset EDA", fontweight="bold",fontsize=14)
    plt.tight_layout(); save(fig,"FigM_Dataset_EDA")


def print_summary(pv):
    print("\n" + "="*55)
    print("SUMMARY")
    print(f"  Graphs tested      : {len(pv)}")
    print(f"  Avg CGA Speedup    : {pv['Speedup_CGA'].mean():.3f}×")
    print(f"  Avg DTA Speedup    : {pv['Speedup_DTA'].mean():.3f}×")
    print(f"  CGA >1× graphs     : {(pv['Speedup_CGA']>1).sum()} / {len(pv)}")
    print(f"  DTA >1× graphs     : {(pv['Speedup_DTA']>1).sum()} / {len(pv)}")
    print("="*55)
    pv[["topology","n_vertices","BASELINE","CGA","DTA",
        "Speedup_CGA","Speedup_DTA"]].to_csv(os.path.join(OUT_DIR,"speedup_summary.csv"))


def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  CGA-2: Paper Plots (basic CGA)                  ║")
    print("╚══════════════════════════════════════════════════╝")
    df = load_data()
    pv = compute_pivot(df)

    gen_fig11(df)
    gen_fig12(pv)
    gen_topology_box(pv)
    gen_heatmap(pv)
    gen_scatter_dta_cga(pv)
    gen_sorted_bar(pv)
    gen_run_consistency(df)
    gen_topology_grouped(pv)
    gen_frontier_profile()
    gen_eda(df)
    print_summary(pv)

    print(f"\n✓ Plots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()