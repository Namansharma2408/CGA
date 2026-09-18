# CGA-2: Execution Steps

## Prerequisites

```bash
pip install -r requirements.txt
# (pins numpy/scipy/sklearn/pandas/matplotlib/seaborn/plotly/kaleido/networkx; see P3-5/P3-6)
```

---

## Step 0 — Generate Training Graphs

```bash
python scripts/generate_graphs.py
```
Creates `data/train/*.mtx` and `data/test/*.mtx`.

---

## Step 1 — Run Basic CGA Inference → `bfs_run_summary.csv`

```bash
python scripts/run_inference.py
```
→ Produces: `./bfs_run_summary.csv` (repo root; see NAMING.md/paths.py)

---

## Step 2 — Generate CGA Paper Plots (basic CGA only)

```bash
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
python scripts/download_datasets.py
```
*Downloads directly to `data/realworld_graphs/`.*

### 3.2 Run Inference (Basic CGA)
```bash
./build/Release/cga_bfs --graphs data/realworld_graphs/ --runs 3 --output results_realworld_basic.csv
```
Single-graph equivalent: `./build/Release/cga_bfs <graph.mtx> 0 --mode dta --runs 3 --output out.csv`
(`--help` lists all flags; see P2-1.)

### 3.3 Run Inference (Improved CGA)
```bash
./build/Release/cga_bfs_improved --graphs data/realworld_graphs/ --runs 3 --output results_realworld_improved.csv
```
Canonical improved path is `build/Release/cga_bfs_improved` (see P2-2/SPEC.md).

### 3.4 Generate Analysis Plots
```bash
python scripts/analyze_realworld_graph.py
```

**Output:** `results/realworld_plots/` (RW1–RW7 plots)

---

## ─────────────── IMPROVED VERSION ───────────────

## Step 4 — Train Improved Models

```bash
python improved/scripts/train_models.py   # from repo root (any cwd works; see P2-6)
```
→ Produces: `improved/models_trained/*.pkl` (25-feature DT ensemble, see NAMING.md)

---

## Step 5 — Run Improved Inference → `improved/results/inference_results.csv`

```bash
python improved/scripts/run_inference.py  # from repo root
```
→ Produces: `improved/results/inference_results.csv`
(fallback legacy path `improved/bfs_run_summary.csv` also searched by plots)

---

## Step 6 — Generate Improved vs Basic Comparison Plots

```bash
python improved/scripts/plot_improved.py   # from repo root
```

**Requires:** Step 1 AND Step 5 complete (needs both CSVs).
Script exits non-zero with a clear error if either CSV is missing (P3-5).

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
python improved/scripts/visualize_3d_improved.py   # from repo root
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

> **Do I need to train before inference? (P0-1)**
> No — C++ binaries use hardcoded heuristic rules derived from training;
> `.pkl` files are NOT loaded at runtime. Training calibrates/validates the
> rules and records provenance; re-export/codegen is required to change
> binary behavior. See SPEC.md.

```bash

# ── Basic CGA (all from repo root) ─────────────────────────────────
python scripts/generate_graphs.py       # Step 0 — make train/test graphs
python scripts/train_models.py          # Step 1 — train DT models → models_trained/*.pkl
python scripts/run_inference.py         # Step 2 — C++ BFS (hardcoded rules)
                                        #         → bfs_run_summary.csv
python scripts/plot_basic.py            # Step 3 — plot from CSV (see P2-5)
python scripts/analyze_realworld_graph.py  # Step 4 — real-world graph analysis

# ── Improved CGA (all from repo root) ──────────────────────────────
python improved/scripts/train_models.py          # Step 5 — 25-feature DT models
                                                #         → improved/models_trained/*.pkl
python improved/scripts/run_inference.py         # Step 6 — improved BFS
                                                #         → improved/results/inference_results.csv
python improved/scripts/plot_improved.py         # Step 7 — needs basic + improved CSVs
python improved/scripts/visualize_3d_improved.py # Step 8 — 3D viz (requires Step 6)
```
