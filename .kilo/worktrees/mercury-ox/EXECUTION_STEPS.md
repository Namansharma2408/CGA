# CGA-2: Execution Steps

## Prerequisites

```bash
pip install numpy scipy matplotlib seaborn pandas scikit-learn plotly kaleido
```

---

## Step 0 — Generate Training Graphs

```bash
cd e:\CGA\CGA-2
python scripts/generate_graphs.py
```
Creates `data/train/*.mtx` and `data/test/*.mtx`.

---

## Step 1 — Run Basic CGA Inference → `bfs_run_summary.csv`

```bash
cd e:\CGA\CGA-2
python scripts/run_inference.py
```
→ Produces: `CGA-2/bfs_run_summary.csv`

---

## Step 2 — Generate CGA Paper Plots (basic CGA only)

```bash
cd e:\CGA\CGA-2
python scripts/plot_paper_figures.py
```

**Requires:** Step 1 complete (needs `bfs_run_summary.csv`).

**Output:** `results/paper_plots/`

| File | Description |
|------|-------------|
| `Fig11_BarGraph_Time.png` | Execution time bar (log scale), 3 platforms |
| `Fig12_Speedup_LinePlot.png` | Speedup line plot, all graphs |
| `FigA_Speedup_Topology_Boxplot.png` | Speedup by topology (box) |
| `FigB_Speedup_Heatmap.png` | Speedup heatmap (topology × size) |
| `FigC_DTA_vs_CGA_Scatter.png` | DTA vs CGA speedup scatter |
| `FigD_PerGraph_Speedup_Bar.png` | Per-graph sorted speedup bar |
| `FigE_Run_Consistency_CoV.png` | Run-to-run stability |
| `FigF_Topology_Speedup_Group.png` | Avg speedup per topology |
| `FigL_BFS_Frontier_Profile.png` | Frontier growth simulation |
| `FigM_Dataset_EDA.png` | Graph dataset EDA |
| `speedup_summary.csv` | Per-graph stats |

---

## Step 3 — Real-World Datasets (Download, Inference, Analysis)

### 3.1 Download Large Graphs (SuiteSparse/SNAP)
```bash
cd e:\CGA\CGA-2
python scripts/download_datasets.py
```
*Downloads directly to `data/realworld_graphs/`.*

### 3.2 Run Inference (Basic CGA)
```bash
cd e:\CGA\CGA-2
.\build\Release\cga_bfs.exe --graphs data\realworld_graphs\ --runs 3 --output results_realworld_basic.csv
```

### 3.3 Run Inference (Improved CGA)
```bash
cd e:\CGA\CGA-2\improved
.\build\Release\cga_improved_bfs.exe --graphs ..\data\realworld_graphs\ --runs 3 --output results_realworld_improved.csv
```

### 3.4 Generate Analysis Plots
```bash
cd e:\CGA\CGA-2
python scripts/analyze_realworld_graph.py
```

**Output:** `results/realworld_plots/` (RW1–RW7 plots)

---

## ─────────────── IMPROVED VERSION ───────────────

## Step 4 — Train Improved Models

```bash
cd e:\CGA\CGA-2\improved
python scripts/train_models.py
```
→ Produces: `improved/models/*.pkl` (25-feature DT ensemble)

---

## Step 5 — Run Improved Inference → `improved/bfs_run_summary.csv`

```bash
cd e:\CGA\CGA-2\improved
python scripts/run_inference.py
```
→ Produces: `improved/bfs_run_summary.csv`

---

## Step 6 — Generate Improved vs Basic Comparison Plots

```bash
cd e:\CGA\CGA-2\improved
python scripts/plot_improved.py
```

**Requires:** Step 1 AND Step 5 complete (needs both CSVs).
Script will exit with a clear error if either CSV is missing.

**Output:** `improved/results/comparison_plots/`

| File | Description |
|------|-------------|
| `CMP1_BasicVsImproved_Speedup_Bar.png` | Side-by-side speedup bar |
| `CMP2_SortedSpeedup_AllMethods.png` | Sorted speedup line + improvement fill |
| `CMP3_ImprovementRatio_Topology.png` | Improvement ratio boxplot |
| `CMP4_ImprovementHeatmap.png` | Heatmap: topology × size |
| `CMP5_TimeScatter_CGA.png` | Basic vs improved time scatter |
| `CMP5_TimeScatter_DTA.png` | Basic vs improved DTA scatter |
| `CMP6_Summary_Topology.png` | Summary grouped bar |
| `comparison_summary.csv` | All stats |

---

## Step 7 — 3D Visualizations (Improved)

```bash
cd e:\CGA\CGA-2\improved
python scripts/visualize_3d_improved.py
```

**Requires:** Step 5 complete.

**Output:** `improved/results/3d_viz/`

| File | Description |
|------|-------------|
| `VIZ1_3d_comparison_speedup.html` | Basic vs improved speedup in 3D |
| `VIZ2_bfs_<topology>.html` | BFS traversal coloured by improved kernel |
| `VIZ3_swucb_surface.html` | SW-UCB score surface as f(EMA, window_obs) |
| `*_snapshot.png` | Static matplotlib fallbacks |

Open `.html` files in any browser — no server needed.

---

## Quick Reference: Full Run Order

> **Why train before inference?**
> The C++ binary (`cga_bfs.exe`) loads the `.pkl` decision-tree model files
> at startup to select kernels during BFS. If `.pkl` files are missing,
> the binary falls back to a static heuristic (not the learned model).

```bash
cd e:\CGA\CGA-2

# ── Basic CGA ──────────────────────────────────────────────────────
python scripts/generate_graphs.py       # Step 0 — make train/test graphs
python scripts/train_models.py          # Step 1 — train DT models → models_trained/*.pkl
python scripts/run_inference.py         # Step 2 — C++ binary loads .pkl, runs BFS
                                        #         → bfs_run_summary.csv
python scripts/plot_paper_figures.py    # Step 3 — plot from CSV
python scripts/analyze_realworld_graph.py  # Step 4 — real-world graph analysis

# ── Improved CGA (inside improved/) ─────────────────────────────────
cd improved
python scripts/train_models.py          # Step 5 — train improved 25-feature DT models
                                        #         → improved/models/*.pkl
python scripts/run_inference.py         # Step 6 — C++ improved binary loads .pkl
                                        #         → improved/bfs_run_summary.csv
python scripts/plot_improved.py         # Step 7 — requires CGA-2/bfs_run_summary.csv
                                        #         AND improved/bfs_run_summary.csv
python scripts/visualize_3d_improved.py # Step 8 — 3D viz (requires Step 6)
```
