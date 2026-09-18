# SPEC — Single Spec Table (P2-7)

| Item | Basic | Improved |
|------|-------|----------|
| Feature dim | 19 (Table II) | 25 (19 + 6 new, see FEATURE_SPEC.md) |
| Model depths | M1=3, M2=7, M3=6, M4=8 | same |
| DTA window W | 20 (capped UCB) | 20 (sliding window; code UCB_WINDOW=20; docs W=12 was stale) |
| Threshold lr | t_x 0.05, t_m 0.025, t_lb 0.015, decay 0.99 | t_x 0.05/0.995, t_m 0.025/0.99, t_lb 0.015/0.98, t_var 0.01–0.05 (see inference) |
| Binaries | build/Release/cga_bfs | build/Release/cga_bfs_improved |
| Models out | models_trained/*.pkl | improved/models_trained/*.pkl |
| Train graphs | data/train/*.mtx | same |
| Test graphs | data/test/*.mtx | same |
| Results | bfs_run_summary.csv (Dataset,Platform,Total Time) | improved/results/inference_results.csv (graph,variant,run,total_time_ms) |
| Kernels | 8 canonical (see KERNELS.md) | same |
| CUDA arch | CUDA_ARCH env (default 86) | same |

Training labels are heuristic proxies (simulate_*/heuristic_proxy_*) unless
`collect_training_data.py` measured labels are used (see P0-8).
C++ binaries use hardcoded heuristic rules derived from training; `.pkl` files
are NOT loaded at runtime (see P0-1).
