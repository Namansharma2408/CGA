import os
import sys
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from train_models import collect_training_samples

def evaluate_models():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  CGA-2 Root: Platform Model Evaluation (Original)    ║")
    print("╚══════════════════════════════════════════════════════╝")

    train_graphs = glob.glob(os.path.join(PROJECT_ROOT, "data", "train", "*.mtx"))
    test_graphs = glob.glob(os.path.join(PROJECT_ROOT, "data", "test", "*.mtx"))
    all_graphs = train_graphs + test_graphs

    if not all_graphs:
        print("Error: No graphs found in data/train/ or data/test/")
        return

    print(f"Loading {len(all_graphs)} graph MTX files and extracting Original 19-dim features...")
    X, Y1, Y2, Y3, Y4, W, families = collect_training_samples(all_graphs)

    if len(X) == 0:
        print("Error: No valid features extracted.")
        return

    print(f"Extracted features for {len(X)} graphs.")
    print("Using 70% Train / 30% Test split...")

    models_to_test = {
        "Model 1: Platform": (Y1, 3),
        "Model 2: CPU Selector": (Y2, 7),
        "Model 3: GPU Selector": (Y3, 6),
        "Model 4: Universal": (Y4, 8)
    }

    print("-" * 55)
    print(f"{'Selector Model':<22} | {'Train Acc %':<12} | {'Test Acc %':<12}")
    print("-" * 55)

    for name, (Y, max_depth) in models_to_test.items():
        X_train, X_test, Y_train, Y_test, W_train, _ = train_test_split(
            X, Y, W, test_size=0.30, random_state=42
        )

        clf = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=2,
            class_weight="balanced", random_state=42
        )
        clf.fit(X_train, Y_train, sample_weight=W_train)

        train_acc = clf.score(X_train, Y_train) * 100
        test_acc = clf.score(X_test, Y_test) * 100
        print(f"{name:<22} | {train_acc:11.2f}% | {test_acc:11.2f}%")

    print("-" * 55)
    print("✓ Evaluation Complete.")

if __name__ == "__main__":
    evaluate_models()