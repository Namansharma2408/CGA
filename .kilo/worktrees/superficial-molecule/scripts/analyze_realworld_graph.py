"""
analyze_realworld_graph.py
==========================
Part 2: Analyse a real-world graph (SNAP/KONECT dataset) and generate the
same set of plots as the paper, plus CGA-2-specific analysis plots.

The script:
  1. Downloads (or loads from data/raw/) a real-world graph in .mtx format.
     Default: email-EuAll (SNAP) — a directed social/communication graph
     with ~265k nodes, ~420k edges; representative of practical workloads.
  2. Extracts the 19-dim (or 25-dim improved) feature vector at each simulated
     BFS iteration step.
  3. Runs timing simulations for all CPU/GPU kernels.
  4. Applies the trained 4-model decision tree to predict the optimal kernel.
  5. Generates all paper plots + analysis plots for this REAL graph.

Run from CGA-2 directory:
    python scripts/analyze_realworld_graph.py

Outputs -> results/realworld_plots/

NOTES
-----
- If internet is unavailable, place your own .mtx file at data/raw/realworld.mtx
- The script falls back to a large synthetic scale-free graph if no .mtx found.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.gridspec import GridSpec

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR      = os.path.join(PROJECT_ROOT, "results", "realworld_plots")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models_trained")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from feature_extraction import (compute_matrix_stats, build_feature_vector,
                                 extract_all_features_from_graph, FEATURE_NAMES, DIM)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

KERNEL_NAMES = ["PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA", "Gustavson",
                "Sort-Based", "CSR-Vector", "Merge-Based"]
CPU_K = KERNEL_NAMES[:5]
GPU_K = KERNEL_NAMES[5:]
PAL   = {"PM-BHash":"
         "LB-PB-MSPA":"
         "Sort-Based":"


def save(fig, name):
    p = os.path.join(OUT_DIR, name + ".png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")



def load_or_generate_graph():
    """Load real-world graph, fall back to large synthetic."""
    mtx_file = os.path.join(DATA_DIR, "realworld.mtx")
    if os.path.exists(mtx_file):
        print(f"[Load] Reading {mtx_file} ...")
        raw = scipy.io.mmread(mtx_file)
        adj = sp.csr_matrix(raw)
        return adj, "RealWorld (realworld.mtx)"

    snap_url = "https://snap.stanford.edu/data/email-EuAll.txt.gz"
    snap_cache = os.path.join(DATA_DIR, "email-EuAll.txt.gz")
    try:
        import urllib.request, gzip
        print(f"[Download] Fetching email-EuAll from SNAP...")
        urllib.request.urlretrieve(snap_url, snap_cache)
        rows_s, cols_s = [], []
        with gzip.open(snap_cache, "rt") as f:
            for line in f:
                if line.startswith("
                    continue
                a, b = line.split()
                rows_s.append(int(a)); cols_s.append(int(b))
        n = max(max(rows_s), max(cols_s)) + 1
        adj = sp.csr_matrix((np.ones(len(rows_s)), (rows_s, cols_s)), shape=(n, n))
        adj = (adj + adj.T).astype(float)
        sp.save_npz(os.path.join(DATA_DIR, "email_euall.npz"), adj)
        return adj, "email-EuAll (SNAP)"
    except Exception as e:
        print(f"  [warn] Could not download: {e}")

    print("[fallback] Generating large synthetic scale-free graph (n=50000)...")
    from scripts.generate_graphs import create_scale_free_graph_mtx
    try:
        A, lbl = create_scale_free_graph_mtx(50000)
        return A, lbl
    except Exception:
        print("[fallback-2] Using random sparse graph n=30000...")
        n = 30000
        rng = np.random.RandomState(42)
        nnz = n * 20
        rows_r = rng.randint(0, n, nnz)
        cols_r = rng.randint(0, n, nnz)
        adj = sp.csr_matrix((np.ones(nnz), (rows_r, cols_r)), shape=(n, n))
        return adj, "Random sparse (n=30k fallback)"



def simulate_kernel_times(feat, ms):
    xd  = feat[10]; md = feat[12]
    n   = ms["n"];  nnz = ms["nnz"]; ad = ms["avg_degree"]
    ae_x = max(1.0, n * xd * ad); x_nnz = max(1.0, n * xd)
    ae_m = max(1.0, n * md * ad)

    t = {}
    t["PM-BHash"]    = ae_x * 2.0 + n * 0.1
    t["LB-PM-BHash"] = t["PM-BHash"] + x_nnz * 0.5
    t["PB-MSPA"]     = ae_m * 1.5 + n * 0.2
    t["LB-PB-MSPA"]  = t["PB-MSPA"]  + x_nnz * 0.5
    t["Gustavson"]   = n * 0.5 + ae_m * 1.0
    if xd > 0.05:
        t["PM-BHash"] *= 1.5 + 5*xd; t["LB-PM-BHash"] *= 1.3 + 4*xd
    if md > 0.4:
        t["PB-MSPA"] *= 0.6; t["LB-PB-MSPA"] *= 0.6

    gpu_oh = 3000.0
    t["Sort-Based"]  = ae_x * 0.8 + gpu_oh
    t["CSR-Vector"]  = nnz * 0.2 + gpu_oh + 1000.0
    t["Merge-Based"] = nnz * 0.15 + gpu_oh + 2000.0
    if xd > 0.05: t["Sort-Based"] *= 2.0 + 10*xd
    return t


def best_kernel(times):
    return min(times, key=times.get)



def load_models():
    import pickle, glob
    pkls = glob.glob(os.path.join(MODEL_DIR, "*.pkl"))
    models = {}
    for p in pkls:
        name = os.path.splitext(os.path.basename(p))[0]
        with open(p, "rb") as f:
            models[name] = pickle.load(f)
    return models


def predict_platform(models, feat):
    if "model1_platform" not in models:
        return "CPU"
    return ["CPU","GPU"][models["model1_platform"].predict([feat])[0]]


def predict_cpu_kernel(models, feat):
    cpu_k = ["PM-BHash","LB-PM-BHash","PB-MSPA","LB-PB-MSPA","Gustavson"]
    if "model2_cpu" not in models:
        return cpu_k[0]
    return cpu_k[models["model2_cpu"].predict([feat])[0]]



def analyze_and_plot(adj, graph_name):
    ms   = compute_matrix_stats(adj)
    rows = extract_all_features_from_graph(adj)
    D    = len(rows)

    print(f"\n  Graph : {graph_name}")
    print(f"  n     : {ms['n']:,}")
    print(f"  nnz   : {ms['nnz']:,}")
    print(f"  avg_d : {ms['avg_degree']:.2f}")
    print(f"  gini  : {ms['gini_row']:.4f}")
    print(f"  BFS depths: {D}")

    depths, xds, mds, best_cpus, best_gpus = [], [], [], [], []
    kernel_times_all = []
    models = load_models()

    for feat, x_nnz, m_nnz, d in rows:
        t = simulate_kernel_times(feat, ms)
        depths.append(d)
        xds.append(feat[10]); mds.append(feat[12])
        best_cpus.append(min(t[k] for k in CPU_K))
        best_gpus.append(min(t[k] for k in GPU_K))
        kernel_times_all.append(t)

    print("\n  [RW-1] Feature heatmap across BFS depths...")
    feat_matrix = np.array([r[0] for r in rows])
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(feat_matrix.T, aspect="auto", cmap="RdYlGn")
    ax.set_yticks(range(len(FEATURE_NAMES[:DIM])))
    ax.set_yticklabels(FEATURE_NAMES[:DIM], fontsize=7)
    ax.set_xlabel("BFS Depth")
    ax.set_title(f"RW-1: Feature Vector Evolution — {graph_name}", fontweight="bold")
    plt.colorbar(im, ax=ax)
    save(fig, "RW1_Feature_Heatmap")

    print("  [RW-2] Frontier density profile...")
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.plot(depths, xds, "b-o", lw=2, ms=4, label="x_density (frontier)")
    ax1.plot(depths, mds, "g-s", lw=2, ms=4, label="m_density (mask)")
    ax1.set_ylabel("Density"); ax1.set_xlabel("BFS Depth")
    ax2.plot(depths, [best_cpus[i]/(best_cpus[i]+best_gpus[i]) for i in range(len(depths))],
             "r--", lw=1.5, label="CPU advantage ratio")
    ax2.set_ylabel("CPU advantage (CPU_cost/total)", color="red")
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, l1+l2, fontsize=8)
    ax1.set_title(f"RW-2: BFS Density Profile — {graph_name}", fontweight="bold")
    save(fig, "RW2_Density_Profile")

    print("  [RW-3] Kernel time per BFS depth...")
    fig, ax = plt.subplots(figsize=(12, 5))
    for k in CPU_K:
        times = [kernel_times_all[i][k] for i in range(len(depths))]
        ax.plot(depths, times, "o-", color=PAL.get(k,"
    ax.set_xlabel("BFS Depth"); ax.set_ylabel("Simulated Time (µs)")
    ax.set_title(f"RW-3: CPU Kernel Time at Each BFS Depth — {graph_name}", fontweight="bold")
    ax.legend(fontsize=8); ax.set_yscale("log")
    save(fig, "RW3_KernelTimes_per_Depth")

    print("  [RW-4] Optimal kernel distribution...")
    opt_counts = {}
    for t in kernel_times_all:
        k = best_kernel(t)
        opt_counts[k] = opt_counts.get(k, 0) + 1
    fig, ax = plt.subplots(figsize=(7, 7))
    wedge_colors = [PAL.get(k, "
    ax.pie(list(opt_counts.values()), labels=list(opt_counts.keys()),
           autopct="%1.1f%%", colors=wedge_colors, startangle=90,
           textprops={"fontsize": 9})
    ax.set_title(f"RW-4: Optimal Kernel Distribution — {graph_name}", fontweight="bold")
    save(fig, "RW4_OptimalKernel_Pie")

    print("  [RW-5] Model prediction vs optimal kernel...")
    if models:
        predicted = []
        optimal   = []
        for feat, x_nnz, m_nnz, d in rows:
            t   = simulate_kernel_times(feat, ms)
            opt = best_kernel({k: t[k] for k in CPU_K})
            prd = predict_cpu_kernel(models, feat)
            predicted.append(prd); optimal.append(opt)
        correct = sum(p == o for p, o in zip(predicted, optimal))
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(predicted))
        for i, (p, o) in enumerate(zip(predicted, optimal)):
            color = "green" if p==o else "red"
            ax.bar(i, 1, color=color, alpha=0.7)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="green", label="Correct"),
                            Patch(color="red",   label="Wrong")], fontsize=9)
        ax.set_xlabel("BFS Depth Index")
        ax.set_title(f"RW-5: CGA Model Predictions vs Optimal — {graph_name}\n"
                     f"Accuracy: {correct}/{len(predicted)} ({100*correct/max(1,len(predicted)):.1f}%)",
                     fontweight="bold")
        ax.axis("off"); ax.set_xlim(-1, len(predicted))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(range(len(optimal)), [CPU_K.index(o) for o in optimal],
                   c="steelblue", label="Optimal", s=30, alpha=0.8)
        ax.scatter(range(len(predicted)), [CPU_K.index(p) for p in predicted],
                   c="tomato", marker="x", label="CGA Predicted", s=30, alpha=0.8)
        ax.set_yticks(range(len(CPU_K))); ax.set_yticklabels(CPU_K, fontsize=8)
        ax.set_xlabel("BFS Depth Index")
        ax.set_title(f"RW-5: CGA Model Predictions vs Optimal — Acc={100*correct/max(1,len(predicted)):.1f}%")
        ax.legend()
        save(fig, "RW5_Prediction_vs_Optimal")
    else:
        print("    [skip] No trained models — skipping RW-5. Run train_models.py first.")

    print("  [RW-6] Estimated speedup vs always-worst strategy...")
    worst = [max(kernel_times_all[i][k] for k in CPU_K) for i in range(len(depths))]
    best  = [min(kernel_times_all[i][k] for k in CPU_K) for i in range(len(depths))]
    speedup = [w/max(b, 1e-9) for w, b in zip(worst, best)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(depths, speedup, color="
    ax.axhline(1.0, color="red", ls="--", lw=1.2, label="Baseline (always worst)")
    ax.set_xlabel("BFS Depth"); ax.set_ylabel("Speedup (best/worst kernel)")
    ax.set_title(f"RW-6: Potential Speedup from Optimal Selection — {graph_name}", fontweight="bold")
    ax.legend()
    save(fig, "RW6_Potential_Speedup")

    print("  [RW-7] x_density vs m_density coloured by optimal kernel...")
    fig, ax = plt.subplots(figsize=(8, 6))
    opt_per_depth = [best_kernel({k: kernel_times_all[i][k] for k in CPU_K})
                     for i in range(len(depths))]
    for k in CPU_K:
        idxs = [i for i, o in enumerate(opt_per_depth) if o==k]
        if idxs:
            ax.scatter([xds[i] for i in idxs], [mds[i] for i in idxs],
                       c=PAL.get(k,"
    ax.set_xlabel("x_density (frontier density)")
    ax.set_ylabel("m_density (mask density)")
    ax.set_title(f"RW-7: Kernel Decision Regions — {graph_name}", fontweight="bold")
    ax.legend(fontsize=8)
    save(fig, "RW7_Kernel_Decision_Regions")

    out_csv = os.path.join(OUT_DIR, "realworld_analysis.csv")
    pd.DataFrame({
        "depth":      depths,
        "x_density":  xds,
        "m_density":  mds,
        "best_cpu_t": best_cpus,
        "best_gpu_t": best_gpus,
        "best_kernel": opt_per_depth,
    }).to_csv(out_csv, index=False)
    print(f"  → {out_csv}")


def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  CGA-2: Real-World Graph Analysis                ║")
    print("╚══════════════════════════════════════════════════╝")
    adj, name = load_or_generate_graph()
    analyze_and_plot(adj, name)
    print(f"\n✓ All real-world plots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()