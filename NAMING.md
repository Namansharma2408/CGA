# NAMING — Project Naming Map (P2-9)

| Concept | Canonical | Aliases / legacy | Notes |
|---------|-----------|------------------|-------|
| Repo | CGA-Accelerated | CGA-2 (paper system name, docs) | Root dir; paper system = CGA-2 |
| Basic binary | build/Release/cga_bfs | build/cga_bfs (CMake fallback) | 19-dim |
| Improved binary | build/Release/cga_bfs_improved | (legacy improved/build/ removed) | 25-dim, see P2-2 |
| Static ML mode | cga / static-ml | STATIC enum (alias STATIC_ML) | Paper static execution flow; not `static` keyword |
| Dynamic mode | dta | DTA_MODE | UCB + adaptive thresholds |
| Baseline | baseline | BASELINE | CPU LB-MSPA (Gustavson) |
| Basic models | models_trained/*.pkl | — | 19-dim, currently heuristic-proxy (P0-1/P0-8) |
| Improved models | improved/models_trained/*.pkl | (legacy improved/models/*.pkl removed) | 25-dim |
| Basic results | bfs_run_summary.csv | results/inference_results.csv (alt) | See paths.py |
| Improved results | improved/results/inference_results.csv | improved/bfs_run_summary.csv (fallback) | See paths.py |
| Plots | scripts/plot_basic.py, improved/scripts/plot_improved.py | scripts/archive/* (legacy) | Max 2 entry points (P2-5) |
