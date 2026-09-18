"""
plot_improved.py
================
Improved CGA — Comparison & Analysis Plots.

WORKFLOW:
  This script MUST be run AFTER inference! The workflow is:

  Step 1: [CGA-2/]   python scripts/run_inference.py
          → produces: CGA-2/bfs_run_summary.csv  (basic CGA results)

  Step 2: [improved/] python scripts/run_inference.py
          → produces: improved/bfs_run_summary.csv  (improved results)

  Step 3: [improved/] python scripts/train_models.py
          → produces: improved/models/*.pkl

  Step 4: [improved/] python scripts/plot_improved.py
          → produces: improved/results/comparison_plots/*.png

If either CSV is missing the script will EXIT EARLY with clear instructions.

Run from CGA-2/ directory:
    python improved/scripts/plot_improved.py

OR from the improved/ directory:
    python scripts/plot_improved.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMPROVED_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR     = os.path.dirname(IMPROVED_DIR)

CSV_BASIC_PRIM    = os.path.join(ROOT_DIR,     "results", "inference_results.csv")
CSV_BASIC_FALL    = os.path.join(ROOT_DIR,     "bfs_run_summary.csv")
CSV_IMPROVED_PRIM = os.path.join(IMPROVED_DIR, "results", "inference_results.csv")
CSV_IMPROVED_FALL = os.path.join(IMPROVED_DIR, "bfs_run_summary.csv")

MODEL_DIR    = os.path.join(IMPROVED_DIR, "models")
OUT_DIR      = os.path.join(IMPROVED_DIR, "results", "comparison_plots")
os.makedirs(OUT_DIR, exist_ok=True)

def find_csv(primary, fallback):
    if os.path.exists(primary): return primary
    if os.path.exists(fallback): return fallback
    return None

path_basic = find_csv(CSV_BASIC_PRIM, CSV_BASIC_FALL)
path_improved = find_csv(CSV_IMPROVED_PRIM, CSV_IMPROVED_FALL)

missing = []
if not path_basic:
    missing.append(f"  MISSING (basic CGA): {CSV_BASIC_PRIM}\n"
                   "  → Run:  cd CGA-2 && python scripts/run_inference.py")
if not path_improved:
    missing.append(f"  MISSING (improved):  {CSV_IMPROVED_PRIM}\n"
                   "  → Run:  cd CGA-2/improved && python scripts/run_inference.py")

if missing:
    print("╔══════════════════════════════════════════════════════╗")
    print("║  ERROR: Inference results not found!                 ║")
    print("╠══════════════════════════════════════════════════════╣")
    for m in missing:
        print(m)
    print("╚══════════════════════════════════════════════════════╝")
    sys.exit(1)

CSV_BASIC = path_basic
CSV_IMPROVED = path_improved

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
BASIC_COLOR = "#58a6ff"
IMP_COLOR   = "#2ea043"
DTA_COLOR   = "#d2a8ff"
BASE_COLOR  = "#8b949e"


def save(fig, name):
    p = os.path.join(OUT_DIR, name + ".png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")



def load_and_merge():
    def prep(path, suffix):
        df = pd.read_csv(path)
        rename_map = {
            "graph": "Dataset",
            "variant": "Platform",
            "total_time_ms": "Total Time (ms)"
        }
        df = df.rename(columns=rename_map)

        if "Dataset" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Dataset"})

        df["topology"]   = df["Dataset"].astype(str).str.extract(r"graph_\d+_([a-z_]+)_V\d+").fillna("unknown")
        df["n_vertices"] = df["Dataset"].astype(str).str.extract(r"_V(\d+)_E").fillna(0).astype(int)

        if "Platform" not in df.columns:
            pass

        mean_df = df.groupby(["Dataset","Platform"])["Total Time (ms)"].mean().reset_index()
        pv = mean_df.pivot(index="Dataset", columns="Platform", values="Total Time (ms)")
        pv.columns = [f"{c}_{suffix}" for c in pv.columns]
        meta = df[["Dataset","topology","n_vertices"]].drop_duplicates("Dataset").set_index("Dataset")
        return pv.join(meta)

    basic = prep(CSV_BASIC,    "basic")
    imp   = prep(CSV_IMPROVED, "imp")
    pv    = basic.join(imp.drop(columns=["topology","n_vertices"]), how="inner")

    ref = pv.get("BASELINE_basic", pv.get("BASELINE_imp"))
    for col in ["CGA_basic","DTA_basic","CGA_imp","DTA_imp"]:
        if col in pv.columns:
            pv[f"Speedup_{col}"] = ref / pv[col]

    if "CGA_basic" in pv.columns and "CGA_imp" in pv.columns:
        pv["Improv_CGA"] = pv["CGA_basic"] / pv["CGA_imp"]
    if "DTA_basic" in pv.columns and "DTA_imp" in pv.columns:
        pv["Improv_DTA"] = pv["DTA_basic"] / pv["DTA_imp"]

    return pv


def gen_cmp1(pv):
    print("[CMP-1] Basic vs Improved speedup bar...")
    cols = [c for c in ["Speedup_CGA_basic","Speedup_DTA_basic",
                         "Speedup_CGA_imp",  "Speedup_DTA_imp"] if c in pv.columns]
    labels_map = {
        "Speedup_CGA_basic": "Basic CGA",
        "Speedup_DTA_basic": "Basic DTA",
        "Speedup_CGA_imp":   "Improved CGA",
        "Speedup_DTA_imp":   "Improved DTA",
    }
    colors_map = {
        "Speedup_CGA_basic": BASIC_COLOR,
        "Speedup_DTA_basic": DTA_COLOR,
        "Speedup_CGA_imp":   IMP_COLOR,
        "Speedup_DTA_imp":   "#2ea043"
    }
    sub    = pv.head(15).reset_index()
    labels = [f"{t[:3].upper()}-{i+1}" for i,t in enumerate(sub["topology"])]
    x = np.arange(len(sub)); n = len(cols)
    width = 0.7 / n

    fig, ax = plt.subplots(figsize=(15, 6))
    for k, col in enumerate(cols):
        if col in sub.columns:
            ax.bar(x + (k - n/2 + 0.5)*width, sub[col], width,
                   color=colors_map[col], label=labels_map[col], alpha=0.85)
    ax.axhline(1.0, color="black", ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Speedup vs Baseline (×)")
    ax.set_title("CMP-1: Basic vs Improved — Speedup Across Graphs", fontweight="bold")
    ax.legend(fontsize=9)
    save(fig, "CMP1_BasicVsImproved_Speedup_Bar")


def gen_cmp2(pv):
    print("[CMP-2] Sorted speedup line (all graphs)...")
    sort_col = "Speedup_DTA_imp" if "Speedup_DTA_imp" in pv.columns else pv.columns[0]
    pv_s = pv.sort_values(sort_col, ascending=False)
    x    = range(len(pv_s))

    fig, ax = plt.subplots(figsize=(14, 5))
    line_styles = {
        "Speedup_CGA_basic": ("-",  BASIC_COLOR, 1.5, "Basic CGA"),
        "Speedup_DTA_basic": ("--", DTA_COLOR,   1.5, "Basic DTA"),
        "Speedup_CGA_imp":   ("-",  IMP_COLOR,   2.2, "Improved CGA"),
        "Speedup_DTA_imp":   ("--", "#2ea043",   2.2, "Improved DTA"),
    }
    for col, (ls, col_c, lw, lbl) in line_styles.items():
        if col in pv_s.columns:
            ax.plot(x, pv_s[col], ls, color=col_c, lw=lw, ms=3, label=lbl)

    if "Speedup_DTA_basic" in pv_s.columns and "Speedup_DTA_imp" in pv_s.columns:
        ax.fill_between(x, pv_s["Speedup_DTA_basic"], pv_s["Speedup_DTA_imp"],
                        alpha=0.12, color=IMP_COLOR, label="Improvement δ (DTA)")

    ax.axhline(1.0, color="black", ls=":", lw=1.0)
    ax.set_xlabel("Graphs (sorted by Improved DTA speedup)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("CMP-2: All-Graph Speedup — Basic vs Improved", fontweight="bold")
    ax.legend(fontsize=9)
    save(fig, "CMP2_SortedSpeedup_AllMethods")


def gen_cmp3(pv):
    print("[CMP-3] Improvement ratio boxplot by topology...")
    rows = []
    for col, label in [("Improv_CGA","CGA"), ("Improv_DTA","DTA")]:
        if col in pv.columns:
            for topo, val in zip(pv["topology"], pv[col]):
                rows.append({"Topology": topo, "Method": label, "Improvement": val})
    if not rows:
        print("  [skip] No improvement ratio columns available."); return
    df_m = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="Topology", y="Improvement", hue="Method", data=df_m,
                palette={"CGA": IMP_COLOR, "DTA": "#d2a8ff"})
    ax.axhline(1.0, color="red", ls="--", lw=1.2, label="No improvement")
    ax.set_title("CMP-3: Improvement Ratio by Topology\n"
                 "(>1 = improved is faster than basic)", fontweight="bold")
    ax.set_ylabel("basic_time / improved_time")
    ax.legend()
    save(fig, "CMP3_ImprovementRatio_Topology")


def gen_cmp4(pv):
    print("[CMP-4] Improvement heatmap...")
    if "Improv_DTA" not in pv.columns:
        print("  [skip] Improv_DTA not available."); return
    pv2 = pv.reset_index()
    pv2["size_bin"] = pd.cut(pv2["n_vertices"],
                              bins=[0,7000,12000,16000,25000],
                              labels=["XS","S","M","L"])
    hm = pv2.groupby(["topology","size_bin"])["Improv_DTA"].mean().unstack(fill_value=np.nan)
    fig, ax = plt.subplots(figsize=(9,5))
    sns.heatmap(hm, annot=True, fmt=".3f", cmap="PuBuGn", ax=ax,
                linewidths=0.5, vmin=1.0)
    ax.set_title("CMP-4: Improvement Ratio Heatmap — Improved vs Basic DTA\n"
                 "(topology × size)", fontweight="bold")
    save(fig, "CMP4_ImprovementHeatmap")


def gen_cmp5(pv):
    print("[CMP-5] Time scatter basic vs improved...")
    for mode, b_col, i_col, color in [
        ("CGA", "CGA_basic", "CGA_imp", BASIC_COLOR),
        ("DTA", "DTA_basic", "DTA_imp", DTA_COLOR),
    ]:
        if b_col not in pv.columns or i_col not in pv.columns:
            continue
        topo_colors = {"scale_free":"#58a6ff", "random":"#2ea043", "regular":"#d2a8ff", "small_world":"#8b949e"}
        cs = [topo_colors.get(t,"#333333") for t in pv["topology"]]
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(pv[b_col], pv[i_col], c=cs, s=55, alpha=0.8, edgecolors="k", lw=0.4)
        lo = min(pv[b_col].min(), pv[i_col].min())
        hi = max(pv[b_col].max(), pv[i_col].max())
        ax.plot([lo,hi],[lo,hi],"k--",lw=1.5,label="Equal performance")
        ax.fill_between([lo,hi],[lo,hi],[lo,lo],alpha=0.06,color="green",label="Improved faster")
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=c, label=t.replace("_"," ").title())
                   for t, c in topo_colors.items()] + \
                  [plt.Line2D([],[],color="k",ls="--",label="Equal")]
        ax.legend(handles=handles, fontsize=8)
        ax.set_xlabel(f"Basic {mode} Time (ms)")
        ax.set_ylabel(f"Improved {mode} Time (ms)")
        ax.set_title(f"CMP-5: {mode} Basic vs Improved — Time Scatter\n"
                     f"(points below diagonal = improved faster)", fontweight="bold")
        save(fig, f"CMP5_TimeScatter_{mode}")


def gen_cmp6(pv):
    print("[CMP-6] Summary grouped bar by topology...")
    sp_cols = [c for c in ["Speedup_CGA_basic","Speedup_DTA_basic",
                            "Speedup_CGA_imp",  "Speedup_DTA_imp"] if c in pv.columns]
    g = pv.groupby("topology")[sp_cols].mean()
    col_labels = {
        "Speedup_CGA_basic": ("Basic CGA", BASIC_COLOR),
        "Speedup_DTA_basic": ("Basic DTA", DTA_COLOR),
        "Speedup_CGA_imp":   ("Improved CGA", IMP_COLOR),
        "Speedup_DTA_imp":   ("Improved DTA", "#2ea043"),
    }
    x = np.arange(len(g))
    n = len(sp_cols); width = 0.7 / n
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, col in enumerate(sp_cols):
        lbl, clr = col_labels[col]
        ax.bar(x + (k - n/2 + 0.5)*width, g[col], width,
               color=clr, label=lbl, alpha=0.85)
        for i, v in enumerate(g[col]):
            ax.text(x[i] + (k - n/2 + 0.5)*width, v + 0.01,
                    f"{v:.2f}×", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.axhline(1.0, color="black", ls="--", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(g.index, fontsize=10)
    ax.set_ylabel("Avg Speedup vs Baseline (×)")
    ax.set_title("CMP-6: Summary — Basic vs Improved by Topology", fontweight="bold")
    ax.legend(fontsize=9)
    save(fig, "CMP6_Summary_Topology")

    out_csv = os.path.join(OUT_DIR, "comparison_summary.csv")
    pv[["topology","n_vertices"] + sp_cols +
       [c for c in ["Improv_CGA","Improv_DTA"] if c in pv.columns]].to_csv(out_csv)
    print(f"  [saved] {out_csv}")

    print("\n  ===== COMPARISON SUMMARY =====")
    for col in sp_cols:
        print(f"  Avg {col_labels[col][0]:18s}: {pv[col].mean():.3f}×")
    for col, lbl in [("Improv_CGA","CGA improvement"),("Improv_DTA","DTA improvement")]:
        if col in pv.columns:
            print(f"  Avg {lbl:18s}: {pv[col].mean():.3f}×")
    print("  " + "="*32)


def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  CGA-2 Improved: Comparison Plots                ║")
    print("║  Reading real inference results from CSV...      ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Basic CSV   : {CSV_BASIC}")
    print(f"  Improved CSV: {CSV_IMPROVED}")

    pv = load_and_merge()
    print(f"  Graphs matched: {len(pv)}")

    gen_cmp1(pv)
    gen_cmp2(pv)
    gen_cmp3(pv)
    gen_cmp4(pv)
    gen_cmp5(pv)
    gen_cmp6(pv)

    print(f"\n✓ All comparison plots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
    print(f"  Basic CSV   : {CSV_BASIC}")
    print(f"  Improved CSV: {CSV_IMPROVED}")

    pv = load_and_merge()
    print(f"  Graphs matched: {len(pv)}")

    gen_cmp1(pv)
    gen_cmp2(pv)
    gen_cmp3(pv)
    gen_cmp4(pv)
    gen_cmp5(pv)
    gen_cmp6(pv)

    print(f"\n✓ All comparison plots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()