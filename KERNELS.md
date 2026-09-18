# KERNELS — Canonical Kernel Ontology (P0-5, P1-2)

| ID | Canonical name | Platform | Implementation | Train label |
|----|----------------|----------|----------------|-------------|
| 0 | PM-BHash | CPU | cpu_pm_bhash | PM-BHash |
| 1 | LB-PM-BHash | CPU | cpu_lb_pm_bhash | LB-PM-BHash |
| 2 | PB-MSPA | CPU | cpu_pb_mspa | PB-MSPA |
| 3 | LB-PB-MSPA | CPU | cpu_lb_pb_mspa | LB-PB-MSPA |
| 4 | LB-MSPA | CPU | cpu_gustavson (Gustavson SpMSpV) | LB-MSPA (legacy alias Gustavson) |
| 5 | Sort-Based SpMSpV | GPU | gpu_sort_based_spmspv | Sort-Based SpMSpV (alias Sort-Based) |
| 6 | SpMV | GPU | gpu_csr_vector_spmv | SpMV (alias CSR-Vector) |
| 7 | Merge-Based SpMV | GPU | gpu_merge_spmv | Merge-Based SpMV (alias Merge-Based) |

Rules:
- C++ `dispatch_kernel` accepts canonical + legacy aliases (see
  `src/core/kernel_registry.h`); unknown names throw with valid-ID list.
- DTA candidate set = all 8 (sort/merge reachable via UCB exploration).
- Greedy DTA emits SpMV vs CPU MSPA/BHash; Model-2 emits 5 CPU names;
  Model-3 emits 3 GPU long names; Model-4 emits LB-MSPA vs SpMV.
- Python training MUST use canonical names (see `scripts/train_models.py`
  CPU_KERNELS/GPU_KERNELS); `test_kernel_registry` asserts every train label
  is dispatchable.
