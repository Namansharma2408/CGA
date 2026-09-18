"""
train_models.py – Training for CGA-BFS 4-model system (19-dim, Table II).

Paper: "Accelerating BFS Through a Sparsity-Aware Adaptive Framework
        on Heterogeneous Platforms"

This script:
  1. Loads .mtx training graphs from data/train/
  2. Extracts real 19-dim features (Table II) via feature_extraction.py.
  3. Computes HEURISTIC-PROXY timing per kernel (heuristic_proxy_*; NOT
     profiled — realistic proxy when GPU unavailable; see P0-8/SPEC.md).
  4. Labels every sample as argmin(proxy_times).
  5. Applies Eq. 6 sample weights + depth-iteration weighting.
  6. Trains 4 decision trees: M1=3, M2=7, M3=6, M4=8 (see SPEC.md).
  7. Saves models as models_trained/*.pkl + provenance.json.

NOTE (P0-1): C++ binaries currently use hardcoded heuristic rules derived
from training; .pkl files are NOT loaded at runtime. Retrain + codegen is
required to change binary behavior (see SPEC.md).

Run:
    python scripts/generate_graphs.py
    python scripts/train_models.py
"""

import os
import sys
import glob
import pickle
import math
import subprocess
import numpy as np
import scipy.io
import scipy.sparse as sp
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
try:
    from scripts.paths import MODELS_TRAINED_DIR, TRAIN_GLOB, TEST_GLOB  # type: ignore
except ImportError:  # fallback when run as scripts/train_models.py with cwd=root
    from paths import MODELS_TRAINED_DIR, TRAIN_GLOB, TEST_GLOB  # type: ignore
from feature_extraction import (
    compute_matrix_stats,
    build_feature_vector,
    extract_all_features_from_graph,
    FEATURE_NAMES,
    DIM,
)

# P0-5: canonical kernel names (see KERNELS.md). "Gustavson" is legacy alias
# for LB-MSPA; GPU use long names (Sort-Based SpMSpV, SpMV, Merge-Based SpMV).
CPU_KERNELS = ["PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA", "LB-MSPA"]
GPU_KERNELS = ["Sort-Based SpMSpV", "SpMV", "Merge-Based SpMV"]
LEGACY_ALIASES = {"Gustavson": "LB-MSPA", "Sort-Based": "Sort-Based SpMSpV",
                  "CSR-Vector": "SpMV", "Merge-Based": "Merge-Based SpMV"}

MODEL2_CLASSES = CPU_KERNELS
MODEL3_CLASSES = GPU_KERNELS
MODEL4_CLASSES = ["LB-MSPA", "SpMV"]

BETA_EQ6 = 1.0



# P0-8: heuristic timing proxy (NOT profiled). Renamed from simulate_* to
# heuristic_proxy_* to avoid implying measured times. Use
# improved/scripts/collect_training_data.py for measured ground truth.
def heuristic_proxy_cpu_kernel_times(feat: np.ndarray, ms: dict) -> dict:
    """
    Heuristic proxy for relative execution time (μs) for each CPU kernel.
    NOT profiled — structural heuristic only (see P0-8, SPEC.md).
    """
    xd   = feat[10]
    md   = feat[12]
    n    = ms["n"]
    nnz  = ms["nnz"]
    ad   = ms["avg_degree"]

    active_edges_x = max(1.0, n * xd * ad)
    active_edges_m = max(1.0, n * md * ad)
    x_nnz = max(1.0, n * xd)

    hash_access   = 2.0
    prefix_sum    = 0.5
    spa_init      = 0.5

    t = {}
    t["PM-BHash"]    = active_edges_x * hash_access + n * 0.1
    t["LB-PM-BHash"] = t["PM-BHash"] + x_nnz * prefix_sum
    t["PB-MSPA"]     = active_edges_m * 1.5 + n * 0.2
    t["LB-PB-MSPA"]  = t["PB-MSPA"] + x_nnz * prefix_sum
    t["LB-MSPA"]     = n * spa_init + active_edges_m * 1.0  # Gustavson impl

    if xd > 0.05:
        t["PM-BHash"]    *= 1.5 + 5 * xd
        t["LB-PM-BHash"] *= 1.3 + 4 * xd

    if md > 0.4:
        t["PB-MSPA"]    *= 0.6
        t["LB-PB-MSPA"] *= 0.6

    gini = ms["gini_row"]
    if gini < 0.3:
        t["LB-PM-BHash"] *= 1.3
        t["LB-PB-MSPA"]  *= 1.3
    else:
        t["PM-BHash"] *= 1.0 + gini
        t["PB-MSPA"]  *= 1.0 + gini

    return t


def heuristic_proxy_gpu_kernel_times(feat: np.ndarray, ms: dict) -> dict:
    """
    Heuristic proxy for GPU kernel times (NOT profiled; see P0-8).
    """
    xd  = feat[10]
    md  = feat[12]
    n   = ms["n"]
    nnz = ms["nnz"]
    ad  = ms["avg_degree"]

    active_edges_x = max(1.0, n * xd * ad)

    t = {}
    gpu_overhead = 3000.0

    t["Sort-Based SpMSpV"] = active_edges_x * 0.8 + gpu_overhead
    t["SpMV"]              = nnz * 0.2 + gpu_overhead + 1000.0
    t["Merge-Based SpMV"]  = nnz * 0.15 + gpu_overhead + 2000.0

    if xd > 0.05:
        t["Sort-Based SpMSpV"] *= 2.0 + 10 * xd

    if nnz > 50000 and xd > 0.1:
        t["Merge-Based SpMV"] *= 0.8

    return t


# Backward-compat aliases (deprecated; use heuristic_proxy_*).
def simulate_cpu_kernel_times(feat, ms): return heuristic_proxy_cpu_kernel_times(feat, ms)
def simulate_gpu_kernel_times(feat, ms): return heuristic_proxy_gpu_kernel_times(feat, ms)


def simulate_all_kernel_times(feat: np.ndarray, ms: dict) -> np.ndarray:
    """Returns timing array ordered as: CPU (5) + GPU (3), total 8 kernels."""
    cpu_t = heuristic_proxy_cpu_kernel_times(feat, ms)
    gpu_t = heuristic_proxy_gpu_kernel_times(feat, ms)
    return np.array([
        cpu_t["PM-BHash"], cpu_t["LB-PM-BHash"],
        cpu_t["PB-MSPA"],  cpu_t["LB-PB-MSPA"],
        cpu_t["LB-MSPA"],
        gpu_t["Sort-Based SpMSpV"], gpu_t["SpMV"], gpu_t["Merge-Based SpMV"],
    ], dtype=np.float32)



def label_platform(times_8: np.ndarray) -> int:
    """Model 1: 0=CPU, 1=GPU. GPU wins if best GPU time < best CPU time."""
    best_cpu = times_8[:5].min()
    best_gpu = times_8[5:].min()
    return 1 if best_gpu < best_cpu else 0


def label_cpu_kernel(times_8: np.ndarray) -> int:
    """Model 2: argmin over 5 CPU kernels (index 0-4)."""
    return int(np.argmin(times_8[:5]))


def label_gpu_kernel(times_8: np.ndarray) -> int:
    """Model 3: argmin over 3 GPU kernels."""
    return int(np.argmin(times_8[5:]))


def label_universal(times_8: np.ndarray) -> int:
    """Model 4: 0=LB-MSPA (CPU-fallback), 1=SpMV (GPU). Indices: 4=LB-MSPA, 6=SpMV."""
    lb_mspa_t = times_8[4]
    spmv_t    = times_8[6]
    return 1 if spmv_t < lb_mspa_t else 0


def label_graph_family(ms: dict) -> int:
    """Heuristic graph family for StratifiedKFold strata."""
    if ms["is_scale_free"] and ms["avg_degree"] > 10:
        return 0
    elif ms["gini_row"] < 0.25:
        return 1
    else:
        return 2



def weight_eq6(times: np.ndarray, opt_idx: int, beta: float = BETA_EQ6) -> float:
    """
    Paper Eq. 6 sample weight: gives high weight when optimal kernel is
    clearly better than alternatives.

    w = exp( β/(k-1) × Σ_{j≠opt} (t_j/t_opt − 1) )
    """
    t_opt = times[opt_idx]
    if t_opt <= 0:
        return 1.0
    k = len(times)
    gain = sum(min(15.0, times[j] / t_opt - 1.0) for j in range(k) if j != opt_idx)
    exponent = min(20.0, beta / (k - 1) * gain)
    return float(np.exp(exponent))


def depth_weight(d: int, D: int) -> float:
    """
    Depth-of-iteration weighting (html-doc formula):
    High weight at early (platform selection) and late (one-copy) depths.
    """
    return 1.0 + 2.0 * math.exp(-d / 3.0) + 2.0 * math.exp(-(D - d) / 3.0)



def collect_training_samples(graphs: list):
    """
    For each graph, extract feature rows and compute timing labels.
    Returns X, Y1, Y2, Y3, Y4, W, families for stratified splitting.
    """
    X, Y1, Y2, Y3, Y4, W, families = [], [], [], [], [], [], []

    for gpath in graphs:
        try:
            raw = scipy.io.mmread(gpath)
            adj = sp.csr_matrix(raw)
        except Exception as e:
            print(f"  [warn] skipping {gpath}: {e}")
            continue

        ms = compute_matrix_stats(adj)
        if ms["n"] < 4:
            continue

        rows = extract_all_features_from_graph(adj)
        D = len(rows)
        family = label_graph_family(ms)

        for feat, x_nnz, m_nnz, d in rows:
            times_8 = simulate_all_kernel_times(feat, ms)

            y1 = label_platform(times_8)
            y2 = label_cpu_kernel(times_8)
            y3 = label_gpu_kernel(times_8)
            y4 = label_universal(times_8)

            w_cpu = weight_eq6(times_8[:5], y2)
            w_combined = (w_cpu + weight_eq6(times_8[5:], y3)) / 2.0
            w_combined *= depth_weight(d, D)

            X.append(feat)
            Y1.append(y1)
            Y2.append(y2)
            Y3.append(y3)
            Y4.append(y4)
            W.append(w_combined)
            families.append(family)

    return (np.array(X, dtype=np.float32),
            np.array(Y1), np.array(Y2), np.array(Y3), np.array(Y4),
            np.array(W, dtype=np.float32),
            np.array(families))


def evaluate_model(clf, X_test, Y_test, name: str) -> float:
    if len(X_test) == 0:
        print(f"  -> {name}: no test samples")
        return 0.0
    acc = clf.score(X_test, Y_test)
    print(f"  -> {name} Validation Accuracy: {acc * 100:.1f}%")
    return acc


def _provenance() -> dict:
    """P3-9: record hardware provenance alongside labels."""
    import datetime
    prov = {"date": datetime.datetime.utcnow().isoformat() + "Z"}
    try:
        prov["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        prov["git_commit"] = "unknown"
    try:
        prov["nvcc"] = subprocess.check_output(
            ["nvcc", "--version"], text=True).strip().splitlines()[-1]
    except Exception:
        prov["nvcc"] = "unknown"
    prov["cuda_arch"] = os.environ.get("CUDA_ARCH", "86")
    try:
        prov["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True).strip()
    except Exception:
        prov["gpu"] = "unknown"
    return prov


def train_decision_trees():
    os.makedirs(MODELS_TRAINED_DIR, exist_ok=True)

    train_graphs = glob.glob(TRAIN_GLOB)
    test_graphs  = glob.glob(TEST_GLOB)

    if not train_graphs:
        print("Error: No graphs found in data/train/. Run generate_graphs.py first.")
        sys.exit(1)

    print(f"[1/5] Extracting features from {len(train_graphs)} training graphs...")
    X, Y1, Y2, Y3, Y4, W, families = collect_training_samples(train_graphs)

    if len(X) == 0:
        print("Error: Could not extract any features from training graphs.")
        sys.exit(1)

    print(f"      Train samples: {len(X)}  |  Feature dim: {X.shape[1]}")
    print(f"      Platform labels  — CPU:{(Y1==0).sum()}, GPU:{(Y1==1).sum()}")
    print(f"      CPU kernel label distribution: {np.bincount(Y2)}")
    print(f"      GPU kernel label distribution: {np.bincount(Y3)}")

    print(f"\n[2/5] Extracting features from {len(test_graphs)} test graphs...")
    if test_graphs:
        Xt, Y1t, Y2t, Y3t, Y4t, Wt, _ = collect_training_samples(test_graphs)
        print(f"      Test samples: {len(Xt)}")
    else:
        Xt = np.zeros((0, DIM), dtype=np.float32)
        Y1t = Y2t = Y3t = Y4t = np.array([])
        print("      No test graphs found — skipping validation.")

    print("\n[3/5] Training 4 Decision Tree models...")
    models = {
        "model1_platform":  DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=3, class_weight="balanced"),
        "model2_cpu":       DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=2, class_weight="balanced"),
        "model3_gpu":       DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=2, class_weight="balanced"),
        "model4_universal": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=2, class_weight="balanced"),
    }

    label_map = {
        "model1_platform":  (Y1, Y1t),
        "model2_cpu":       (Y2, Y2t),
        "model3_gpu":       (Y3, Y3t),
        "model4_universal": (Y4, Y4t),
    }

    print("\n[4/5] Validation...")
    for name, clf in models.items():
        Y_train, Y_test_arr = label_map[name]
        clf.fit(X, Y_train, sample_weight=W)
        if len(Xt) > 0:
            evaluate_model(clf, Xt, Y_test_arr, name)

    print("\n[5/5] Stratified 5-fold cross-validation for Model 1 (platform)...")
    skf = StratifiedKFold(n_splits=min(5, len(np.unique(families))),
                          shuffle=True, random_state=42)
    fold_accs = []
    m1_proto = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3,
                                       class_weight="balanced")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, families)):
        m1_proto.fit(X[tr_idx], Y1[tr_idx], sample_weight=W[tr_idx])
        acc = m1_proto.score(X[val_idx], Y1[val_idx])
        fold_accs.append(acc)
        print(f"  Fold {fold+1}: {acc*100:.1f}%")
    print(f"  Mean CV Accuracy: {np.mean(fold_accs)*100:.1f}% "
          f"(±{np.std(fold_accs)*100:.1f}%)")

    print("\nExporting models...")
    for name, clf in models.items():
        # P0-4: fail loudly on DIM mismatch (train/serve skew guard).
        assert clf.n_features_in_ == DIM, (
            f"{name}: trained dim {clf.n_features_in_} != FEATURE_DIM {DIM} "
            f"({FEATURE_NAMES}); see FEATURE_SPEC.md")
        path = os.path.join(MODELS_TRAINED_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(clf, f)
        print(f"  -> {path}  [depth={clf.get_depth()}, "
              f"leaves={clf.get_n_leaves()}, "
              f"features={clf.n_features_in_}]")
    import json
    with open(os.path.join(MODELS_TRAINED_DIR, "provenance.json"), "w") as f:
        json.dump({"provenance": _provenance(), "dim": DIM,
                   "features": FEATURE_NAMES,
                   "note": "heuristic_proxy labels (NOT profiled); C++ uses hardcoded rules (P0-1/P0-8)"}, f, indent=2)

    print("\n── Top-5 Feature Importances (Model 1 — Platform Selector) ──")
    importances = models["model1_platform"].feature_importances_
    top5 = np.argsort(importances)[::-1][:5]
    for rank, i in enumerate(top5, 1):
        fname = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"f[{i}]"
        print(f"\n  Feature {rank}: {fname} = {importances[i]:.4f}")

    print(f"\n✓ Successfully trained 4 models on {len(X)} samples "
          f"from {len(train_graphs)} graphs.")
    print("  Models saved to models_trained/")


if __name__ == "__main__":
    train_decision_trees()