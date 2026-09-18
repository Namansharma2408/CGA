"""
visualize_3d_improved.py
========================
3D visualizations for the IMPROVED CGA framework only.

WORKFLOW — run AFTER inference:
  Step 1: [CGA-2/improved/] python scripts/run_inference.py
  Step 2: [CGA-2/improved/] python scripts/visualize_3d_improved.py

Generates interactive Plotly HTML and static PNG snapshots comparing
how the improved algorithm selects kernels across BFS traversal in 3D space.

Outputs -> improved/results/3d_viz/
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMPROVED_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR     = os.path.dirname(IMPROVED_DIR)
OUT_DIR      = os.path.join(IMPROVED_DIR, "results", "3d_viz")
RESULTS_DIR  = os.path.join(IMPROVED_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_BASIC    = os.path.join(ROOT_DIR,     "bfs_run_summary.csv")
CSV_IMPROVED = os.path.join(IMPROVED_DIR, "bfs_run_summary.csv")

if not os.path.exists(CSV_IMPROVED):
    print("╔══════════════════════════════════════════════════╗")
    print("║  ERROR: improved/bfs_run_summary.csv not found!  ║")
    print("║  Run first:                                      ║")
    print("║    cd CGA-2/improved                             ║")
    print("║    python scripts/run_inference.py               ║")
    print("╚══════════════════════════════════════════════════╝")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
    print("[warn] plotly not found. Run: pip install plotly")
    print("       Generating matplotlib snapshots only.")

KERNEL_NAMES  = ["PM-BHash","LB-PM-BHash","PB-MSPA","LB-PB-MSPA","Gustavson"]
KERNEL_COLORS = {
    "PM-BHash":    "
    "LB-PM-BHash": "
    "PB-MSPA":     "
    "LB-PB-MSPA":  "
    "Gustavson":   "
}
IMPROVED_COLOR = "
BASIC_COLOR    = "


def save_html(fig, name):
    p = os.path.join(OUT_DIR, name + ".html")
    fig.write_html(p)
    print(f"  [html] {p}")


def save_png(fig, name):
    p = os.path.join(OUT_DIR, name + ".png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [png]  {p}")


def viz1_comparison_3d():
    print("\n[VIZ-1] 3D speedup space: basic vs improved...")
    if not os.path.exists(CSV_BASIC):
        print("  [skip] basic CSV not found.")
        return

    def load_speedups(path, suffix):
        df = pd.read_csv(path)
        df["topology"]   = df["Dataset"].str.extract(r"graph_\d+_([a-z_]+)_V\d+")
        df["n_vertices"] = df["Dataset"].str.extract(r"_V(\d+)_E").astype(int)
        mean_df = df.groupby(["Dataset","Platform"])["Total Time (ms)"].mean().reset_index()
        pv = mean_df.pivot(index="Dataset", columns="Platform", values="Total Time (ms)")
        for c in ["BASELINE","CGA","DTA"]:
            if c not in pv.columns: pv[c] = np.nan
        pv["Speedup_CGA"] = pv["BASELINE"] / pv["CGA"]
        pv["Speedup_DTA"] = pv["BASELINE"] / pv["DTA"]
        meta = df[["Dataset","topology","n_vertices"]].drop_duplicates("Dataset").set_index("Dataset")
        return pv.join(meta)

    basic = load_speedups(CSV_BASIC,    "basic")
    imp   = load_speedups(CSV_IMPROVED, "imp")
    merged = basic[["Speedup_CGA","Speedup_DTA","topology","n_vertices"]].join(
             imp[["Speedup_CGA","Speedup_DTA"]].rename(
                 columns={"Speedup_CGA":"Speedup_CGA_imp","Speedup_DTA":"Speedup_DTA_imp"}),
             how="inner")

    topo_color = {"scale_free":"

    if PLOTLY_OK:
        colors = [topo_color.get(t,"
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=merged["Speedup_CGA"],
            y=merged["Speedup_CGA_imp"],
            z=merged["n_vertices"],
            mode="markers",
            marker=dict(size=6, color=colors, opacity=0.85,
                        line=dict(color="white", width=0.3)),
            text=[f"{idx}<br>Topology: {t}<br>Basic: {b:.2f}×<br>Improved: {i:.2f}×<br>Vertices: {v}"
                  for idx, t, b, i, v in zip(
                      merged.index, merged["topology"],
                      merged["Speedup_CGA"], merged["Speedup_CGA_imp"], merged["n_vertices"])],
            hoverinfo="text", name="CGA Comparison"))

        for tn, tc in topo_color.items():
            fig.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode="markers",
                                        marker=dict(size=8,color=tc),
                                        name=tn.replace("_"," ").title()))

        lim = max(merged["Speedup_CGA"].max(), merged["Speedup_CGA_imp"].max())
        fig.add_trace(go.Scatter3d(
            x=[0,lim], y=[0,lim], z=[merged["n_vertices"].mean()]*2,
            mode="lines", line=dict(color="white",width=2,dash="dash"),
            name="Basic = Improved"))

        fig.update_layout(
            title="VIZ-1: 3D Speedup Space — Basic CGA vs Improved CGA<br>"
                  "x=Basic speedup, y=Improved speedup, z=Graph size",
            scene=dict(
                xaxis_title="Basic CGA Speedup (×)",
                yaxis_title="Improved CGA Speedup (×)",
                zaxis_title="
                bgcolor="rgba(15,15,30,1)"),
            paper_bgcolor="rgba(15,15,30,1)",
            font=dict(color="white"), height=750)
        save_html(fig, "VIZ1_3d_comparison_speedup")

    fig3, ax3 = plt.subplots(figsize=(9,7), subplot_kw={"projection":"3d"})
    tmap = {"scale_free":"red","small_world":"blue","erdos_renyi":"green"}
    cs   = [tmap.get(t,"gray") for t in merged["topology"]]
    ax3.scatter(merged["Speedup_CGA"], merged["Speedup_CGA_imp"], merged["n_vertices"],
                c=cs, s=30, alpha=0.75, edgecolors="k", lw=0.3)
    lo = min(merged["Speedup_CGA"].min(), merged["Speedup_CGA_imp"].min())
    hi = max(merged["Speedup_CGA"].max(), merged["Speedup_CGA_imp"].max())
    ax3.plot([lo,hi],[lo,hi],[merged["n_vertices"].mean()]*2,"k--",lw=1.5)
    ax3.set_xlabel("Basic CGA Speedup"); ax3.set_ylabel("Improved Speedup")
    ax3.set_zlabel("
    ax3.set_title("Basic vs Improved — 3D Speedup")
    save_png(fig3, "VIZ1_3d_comparison_snapshot")


def force_layout_3d(n, edges, n_iters=60, seed=42):
    rng = np.random.RandomState(seed)
    pos = rng.randn(n, 3) * 3.0
    adj = np.zeros((n, n))
    for u, v in edges:
        adj[u,v] = adj[v,u] = 1.0
    for _ in range(n_iters):
        force = np.zeros((n, 3))
        for i in range(n):
            diff = pos[i] - pos
            dist = np.linalg.norm(diff, axis=1, keepdims=True).clip(0.01)
            force[i] += (diff / dist**3).sum(0) * 0.05
            nbrs = np.where(adj[i] > 0)[0]
            if len(nbrs):
                force[i] -= (pos[i] - pos[nbrs]).mean(0) * 0.15
        pos += force.clip(-0.5, 0.5)
    pos -= pos.mean(0)
    pos /= max(np.abs(pos).max(), 1e-6) / 5.0
    return pos


def make_small_graph(n=60, kind="scale_free"):
    rng = np.random.RandomState(42)
    if kind == "scale_free":
        degrees = (rng.zipf(2.2, n) % (n//3) + 1).astype(int)
        edges = []
        for i in range(n):
            for j in rng.choice(n, min(degrees[i], 4), replace=False):
                if i != j: edges.append((i, j))
    elif kind == "small_world":
        edges = [(i,(i+1)%n) for i in range(n)] + [(i,(i+2)%n) for i in range(n)]
        for k in range(len(edges)):
            if rng.rand() < 0.12: edges[k] = (edges[k][0], rng.randint(0,n))
    else:
        edges = [(i,j) for i in range(n) for j in range(i+1,n) if rng.rand()<0.07]
    return n, list(set(edges))


def bfs_layers(n, edges):
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    visited = [-1]*n; visited[0] = 0
    q = deque([0]); depth = {}; depth[0] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if visited[v] == -1:
                visited[v] = depth[u]+1
                depth[v]   = depth[u]+1
                q.append(v)
    return visited, depth


def kernel_improved_for_depth(d, max_d):
    """Mock improved kernel selection based on BFS depth."""
    frac = d / max(max_d, 1)
    if frac < 0.1:   return "PM-BHash"
    if frac < 0.3:   return "LB-PM-BHash"
    if frac < 0.6:   return "PB-MSPA"
    if frac < 0.85:  return "LB-PB-MSPA"
    return "Gustavson"


def viz2_bfs_graph():
    print("\n[VIZ-2] 3D BFS traversal — improved kernel per node...")
    for kind in ["scale_free", "small_world", "erdos_renyi"]:
        n, edges = make_small_graph(60, kind)
        pos      = force_layout_3d(n, edges)
        visited, depth = bfs_layers(n, edges)
        max_d = max(v for v in visited if v >= 0)

        node_kernels = [kernel_improved_for_depth(depth.get(i,-1), max_d)
                        for i in range(n)]
        node_colors  = [KERNEL_COLORS.get(k,"

        adj_dict = {}
        for u,v in edges:
            adj_dict.setdefault(u,[]).append(v)
            adj_dict.setdefault(v,[]).append(u)
        edge_x, edge_y, edge_z = [], [], []
        for u, nbrs in adj_dict.items():
            for v in nbrs:
                edge_x += [pos[u,0],pos[v,0],None]
                edge_y += [pos[u,1],pos[v,1],None]
                edge_z += [pos[u,2],pos[v,2],None]

        if PLOTLY_OK:
            etrace = go.Scatter3d(x=edge_x,y=edge_y,z=edge_z,mode="lines",
                                   line=dict(color="
                                   hoverinfo="none",name="Edges")
            ntrace = go.Scatter3d(
                x=pos[:,0],y=pos[:,1],z=pos[:,2],mode="markers",
                marker=dict(size=8,color=node_colors,opacity=0.9,
                            line=dict(color="black",width=0.4)),
                text=[f"Node {i}<br>Depth: {visited[i]}<br>Kernel: {node_kernels[i]}"
                      for i in range(n)],
                hoverinfo="text", name="Nodes")
            legend_traces = [
                go.Scatter3d(x=[None],y=[None],z=[None],mode="markers",
                             marker=dict(size=8,color=c),name=k)
                for k,c in KERNEL_COLORS.items()]

            fig = go.Figure(data=[etrace,ntrace]+legend_traces)
            fig.update_layout(
                title=f"VIZ-2: BFS Traversal — {kind.replace('_',' ').title()}"
                      f"<br>Nodes coloured by Improved CGA kernel selection",
                scene=dict(
                    xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    zaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    bgcolor="rgba(15,15,30,1)"),
                paper_bgcolor="rgba(15,15,30,1)",
                font=dict(color="white"), height=700,
                legend=dict(x=1.02,y=0.9))
            save_html(fig, f"VIZ2_bfs_{kind}_improved")

        fig3, ax3 = plt.subplots(figsize=(8,7), subplot_kw={"projection":"3d"})
        for u, nbrs in adj_dict.items():
            for v in nbrs:
                ax3.plot([pos[u,0],pos[v,0]],[pos[u,1],pos[v,1]],[pos[u,2],pos[v,2]],
                         color="
        depths_arr = np.array([visited[i] for i in range(n)])
        ax3.scatter(pos[:,0],pos[:,1],pos[:,2],c=depths_arr,cmap="plasma",
                    s=40,alpha=0.85,edgecolors="k",lw=0.3)
        ax3.set_title(f"Improved BFS: {kind} (coloured by BFS depth)")
        ax3.set_xticklabels([]); ax3.set_yticklabels([]); ax3.set_zticklabels([])
        save_png(fig3, f"VIZ2_bfs_{kind}_snapshot")


def viz3_swucb_surface():
    print("\n[VIZ-3] 3D SW-UCB score surface...")
    ema_vals  = np.linspace(0.5, 20, 40)
    wobs_vals = np.linspace(1, 20, 40)
    EMA, WOBS = np.meshgrid(ema_vals, wobs_vals)
    c     = 2.0; W = 20
    Z_ucb = EMA - c * np.sqrt(np.log(np.minimum(30, W)) / np.maximum(WOBS, 1))

    if PLOTLY_OK:
        fig = go.Figure(data=[go.Surface(
            z=Z_ucb, x=EMA, y=WOBS,
            colorscale="Plasma",
            colorbar=dict(title="UCB Score (lower=preferred)", tickfont=dict(color="white")),
            hovertemplate="EMA=%{x:.1f}ms<br>Window obs=%{y:.0f}<br>UCB=%{z:.2f}<extra></extra>")])
        fig.update_layout(
            title="VIZ-3: Sliding-Window UCB Score Surface<br>"
                  "(lower score → kernel preferred; rewards unexplored & fast kernels)",
            scene=dict(
                xaxis_title="EMA time (ms)",
                yaxis_title="Window observations",
                zaxis_title="UCB Score",
                bgcolor="rgba(15,15,30,1)"),
            paper_bgcolor="rgba(15,15,30,1)",
            font=dict(color="white"), height=700)
        save_html(fig, "VIZ3_swucb_surface")

    fig3, ax3 = plt.subplots(figsize=(9,7), subplot_kw={"projection":"3d"})
    surf = ax3.plot_surface(EMA,WOBS,Z_ucb,cmap="plasma",alpha=0.85,edgecolor="none")
    plt.colorbar(surf,ax=ax3,label="SW-UCB Score",shrink=0.5)
    ax3.set_xlabel("EMA time (ms)"); ax3.set_ylabel("Window observations"); ax3.set_zlabel("UCB Score")
    ax3.set_title("Improved CGA: SW-UCB Score Surface")
    save_png(fig3, "VIZ3_swucb_surface_snapshot")


def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  CGA-2 Improved: 3D Visualizations               ║")
    print("╚══════════════════════════════════════════════════╝")
    if not PLOTLY_OK:
        print("  pip install plotly  to get interactive HTML output\n")

    viz1_comparison_3d()
    viz2_bfs_graph()
    viz3_swucb_surface()

    print(f"\n✓ All 3D plots saved to: {OUT_DIR}/")
    if PLOTLY_OK:
        print("  Open .html files in any browser for interactive exploration.")


if __name__ == "__main__":
    main()