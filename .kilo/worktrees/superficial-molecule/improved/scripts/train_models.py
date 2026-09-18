"""
train_models.py (IMPROVED) – 25-dim feature training for CGA-BFS.

Trains 4 Decision Tree models using the improved 25-dim features.
Saves .pkl models to: CGA-2/improved/models_trained/

Run from CGA-2/improved/ or CGA-2/:
    python improved/scripts/train_models.py

Expects graphs at: CGA-2/data/train/*.mtx and CGA-2/data/test/*.mtx
"""

import os, sys, glob, pickle, math
import numpy as np
import scipy.io
import scipy.sparse as sp
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMPROVED_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(IMPROVED_DIR)

sys.path.insert(0, SCRIPT_DIR)
from feature_extraction import (
    compute_matrix_stats,
    build_feature_vector,
    extract_all_features_from_graph,
    FEATURE_NAMES,
    DIM,
)

MODELS_OUT_DIR = os.path.join(IMPROVED_DIR, "models_trained")

TRAIN_GLOB = os.path.join(PROJECT_ROOT, "data", "train", "*.mtx")
TEST_GLOB  = os.path.join(PROJECT_ROOT, "data", "test",  "*.mtx")

CPU_KERNELS = ["PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA", "Gustavson"]
GPU_KERNELS = ["Sort-Based", "CSR-Vector", "Merge-Based"]
BETA_EQ6 = 1.0



def simulate_cpu_kernel_times(feat: np.ndarray, ms: dict) -> dict:
    xd   = feat[10]
    md   = feat[12]
    n    = ms["n"];  nnz = ms["nnz"];  ad = ms["avg_degree"]
    imbalance = float(feat[20])
    locality  = float(feat[23])

    active_edges_x = max(1.0, n * xd * ad)
    active_edges_m = max(1.0, n * md * ad)
    x_nnz = max(1.0, n * xd)

    t = {}
    t["PM-BHash"]    = active_edges_x * 2.0 * (1.0 + imbalance) + n * 0.1
    t["LB-PM-BHash"] = t["PM-BHash"] * 0.7 + x_nnz * 0.5
    t["PB-MSPA"]     = active_edges_m * 1.5 * max(0.1, 1.2 - locality) + n * 0.2
    t["LB-PB-MSPA"]  = t["PB-MSPA"] * 0.8 + x_nnz * 0.5
    t["Gustavson"]   = n * 0.5 + active_edges_m * 1.0

    if xd > 0.05:
        t["PM-BHash"]    *= 1.5 + 5 * xd
        t["LB-PM-BHash"] *= 1.3 + 4 * xd
    if md > 0.4:
        t["PB-MSPA"]    *= 0.6
        t["LB-PB-MSPA"] *= 0.6

    gini = ms["gini_row"]
    if gini < 0.3:
        t["LB-PM-BHash"] *= 1.3;  t["LB-PB-MSPA"] *= 1.3
    else:
        t["PM-BHash"] *= 1.0 + gini;  t["PB-MSPA"] *= 1.0 + gini
    return t


def simulate_gpu_kernel_times(feat: np.ndarray, ms: dict) -> dict:
    xd  = feat[10];  n = ms["n"];  nnz = ms["nnz"];  ad = ms["avg_degree"]
    launch_penalty = float(feat[24])
    active_edges_x = max(1.0, n * xd * ad)
    gpu_overhead = 3000.0 * (1.0 + launch_penalty)
    t = {
        "Sort-Based":  active_edges_x * 0.8 + gpu_overhead,
        "CSR-Vector":  nnz * 0.2 + gpu_overhead + 1000.0,
        "Merge-Based": nnz * 0.15 + gpu_overhead + 2000.0,
    }
    if xd > 0.05:
        t["Sort-Based"] *= 2.0 + 10 * xd
    if nnz > 50000 and xd > 0.1:
        t["Merge-Based"] *= 0.8
    return t


def simulate_all_kernel_times(feat, ms):
    cpu_t = simulate_cpu_kernel_times(feat, ms)
    gpu_t = simulate_gpu_kernel_times(feat, ms)
    return np.array([
        cpu_t["PM-BHash"], cpu_t["LB-PM-BHash"],
        cpu_t["PB-MSPA"],  cpu_t["LB-PB-MSPA"],
        cpu_t["Gustavson"],
        gpu_t["Sort-Based"], gpu_t["CSR-Vector"], gpu_t["Merge-Based"],
    ], dtype=np.float32)


def label_platform(t8):  return 1 if t8[5:].min() < t8[:5].min() else 0
def label_cpu_kernel(t8): return int(np.argmin(t8[:5]))
def label_gpu_kernel(t8): return int(np.argmin(t8[5:]))
def label_universal(t8):  return 1 if t8[6] < t8[4] else 0

def label_graph_family(ms):
    if ms["is_scale_free"] and ms["avg_degree"] > 10: return 0
    elif ms["gini_row"] < 0.25: return 1
    else: return 2

def weight_eq6(times, opt_idx, beta=BETA_EQ6):
    t_opt = times[opt_idx]
    if t_opt <= 0: return 1.0
    k = len(times)
    gain = sum(min(15.0, times[j]/t_opt - 1.0) for j in range(k) if j != opt_idx)
    return float(np.exp(min(20.0, beta / (k-1) * gain)))

def depth_weight(d, D):
    return 1.0 + 2.0 * math.exp(-d/3.0) + 2.0 * math.exp(-(D-d)/3.0)


def collect_training_samples(graphs):
    X, Y1, Y2, Y3, Y4, W, families = [], [], [], [], [], [], []
    for gpath in graphs:
        try:
            adj = sp.csr_matrix(scipy.io.mmread(gpath))
        except Exception as e:
            print(f"  [warn] skipping {gpath}: {e}"); continue
        ms = compute_matrix_stats(adj)
        if ms["n"] < 4: continue
        rows = extract_all_features_from_graph(adj)
        D = len(rows)
        family = label_graph_family(ms)
        for feat, x_nnz, m_nnz, d in rows:
            t8 = simulate_all_kernel_times(feat, ms)
            y1, y2, y3, y4 = (label_platform(t8), label_cpu_kernel(t8),
                               label_gpu_kernel(t8), label_universal(t8))
            w = (weight_eq6(t8[:5], y2) + weight_eq6(t8[5:], y3)) / 2.0
            w *= depth_weight(d, D)
            X.append(feat); Y1.append(y1); Y2.append(y2)
            Y3.append(y3); Y4.append(y4); W.append(w); families.append(family)
    return (np.array(X, dtype=np.float32),
            np.array(Y1), np.array(Y2), np.array(Y3), np.array(Y4),
            np.array(W, dtype=np.float32), np.array(families))


def train_decision_trees():
    os.makedirs(MODELS_OUT_DIR, exist_ok=True)

    train_graphs = glob.glob(TRAIN_GLOB)
    test_graphs  = glob.glob(TEST_GLOB)

    if not train_graphs:
        print(f"Error: No graphs in {TRAIN_GLOB}")
        print("Run first: python scripts/generate_graphs.py  (from CGA-2/)")
        sys.exit(1)

    print(f"[1/5] Training graphs: {len(train_graphs)} from {PROJECT_ROOT}/data/train/")
    print(f"      Feature dim: {DIM}")
    X, Y1, Y2, Y3, Y4, W, families = collect_training_samples(train_graphs)
    if len(X) == 0:
        print("Error: No features extracted."); sys.exit(1)

    print(f"      Samples: {len(X)}  |  Platform: CPU={( Y1==0).sum()} GPU={(Y1==1).sum()}")

    print(f"\n[2/5] Test graphs: {len(test_graphs)}")
    if test_graphs:
        Xt, Y1t, Y2t, Y3t, Y4t, Wt, _ = collect_training_samples(test_graphs)
    else:
        Xt = np.zeros((0, DIM), dtype=np.float32)
        Y1t = Y2t = Y3t = Y4t = np.array([])

    print("\n[3/5] Training 4 Decision Trees (25-dim)...")
    models = {
        "model1_platform":  DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, class_weight="balanced"),
        "model2_cpu":       DecisionTreeClassifier(max_depth=7, min_samples_leaf=2, class_weight="balanced"),
        "model3_gpu":       DecisionTreeClassifier(max_depth=6, min_samples_leaf=2, class_weight="balanced"),
        "model4_universal": DecisionTreeClassifier(max_depth=8, min_samples_leaf=2, class_weight="balanced"),
    }
    label_map = {
        "model1_platform": (Y1, Y1t), "model2_cpu": (Y2, Y2t),
        "model3_gpu": (Y3, Y3t), "model4_universal": (Y4, Y4t),
    }

    print("\n[4/5] Fitting & validating...")
    for name, clf in models.items():
        Y_train, Y_test_arr = label_map[name]
        clf.fit(X, Y_train, sample_weight=W)
        if len(Xt) > 0:
            acc = clf.score(Xt, Y_test_arr)
            print(f"  {name}: val_acc={acc*100:.1f}%")

    print("\n[5/5] Stratified 5-fold CV (Model1)...")
    skf = StratifiedKFold(n_splits=min(5, max(2, len(np.unique(families)))),
                          shuffle=True, random_state=42)
    fold_accs = []
    m1 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, class_weight="balanced")
    for fold, (tri, vli) in enumerate(skf.split(X, families)):
        m1.fit(X[tri], Y1[tri], sample_weight=W[tri])
        fold_accs.append(m1.score(X[vli], Y1[vli]))
        print(f"  Fold {fold+1}: {fold_accs[-1]*100:.1f}%")
    print(f"  Mean CV: {np.mean(fold_accs)*100:.1f}% (±{np.std(fold_accs)*100:.1f}%)")

    print(f"\nSaving models to: {MODELS_OUT_DIR}/")
    for name, clf in models.items():
        path = os.path.join(MODELS_OUT_DIR, f"{name}.pkl")
        with open(path, "wb") as fh:
            pickle.dump(clf, fh)
        print(f"  -> {path}  [dim={clf.n_features_in_}, leaves={clf.get_n_leaves()}]")

    print("\n── Top-5 Feature Importances (Model1 Platform) ──")
    imp = models["model1_platform"].feature_importances_
    for rank, i in enumerate(np.argsort(imp)[::-1][:5], 1):
        print(f"

    print(f"\n✓ Done. {len(X)} samples from {len(train_graphs)} graphs.")


if __name__ == "__main__":
    train_decision_trees()