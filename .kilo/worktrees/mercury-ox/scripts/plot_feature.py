import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

FEATURE_NAMES = [
    "log_n", "log_nnz", "log_avg_degree",
    "row_cv", "col_cv", "log_max_row_nnz",
    "gini_row", "gini_col", "is_scale_free",
    "log_x_nnz", "x_density", "log_m_nnz", "m_density", "x_decreasing",
    "log_nnz_ratio", "log_valid_nnz", "degree_x", "degree_x_m",
    "log_masked_valid_nnz",
    "local_density_ratio", "frontier_load_imbalance", "avg_degree_weighted",
    "masked_frontier_growth", "cache_locality_estimate", "gpu_launch_penalty"
]

def load_model_importances(model_dir):
    """Loads feature importances for all available models."""
    importances = {}
    if not os.path.exists(model_dir):
        print(f"Directory {model_dir} does not exist.")
        return importances

    for file in os.listdir(model_dir):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "")
            with open(os.path.join(model_dir, file), 'rb') as f:
                clf = pickle.load(f)
                importances[model_name] = clf.feature_importances_
    return importances

def plot_feature_importances():
    basic_models_dir = os.path.join(PROJECT_ROOT, "models_trained")
    improved_models_dir = os.path.join(PROJECT_ROOT, "improved", "models_trained")

    if os.path.exists(improved_models_dir) and len(os.listdir(improved_models_dir)) > 0:
        model_dir = improved_models_dir
        print(f"Loading IMPROVED models from {model_dir}")
        dim = 25
    elif os.path.exists(basic_models_dir) and len(os.listdir(basic_models_dir)) > 0:
        model_dir = basic_models_dir
        print(f"Loading BASIC models from {model_dir}")
        dim = 19
    else:
        print("ERROR: No trained models found. Please train models first (e.g. python improved/scripts/train_models.py).")
        return

    importances = load_model_importances(model_dir)

    if not importances:
        print("No models found inside the directory.")
        return

    models_to_plot = ["model1_platform", "model2_cpu", "model3_gpu", "model4_universal"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, model_name in enumerate(models_to_plot):
        ax = axes[i]
        if model_name not in importances:
            ax.set_visible(False)
            continue

        imp = importances[model_name]

        feat_names = FEATURE_NAMES[:len(imp)]

        sort_idx = np.argsort(imp)[::-1]
        sorted_imp = imp[sort_idx]
        sorted_names = [feat_names[j] for j in sort_idx]

        top_k = 10
        sorted_imp = sorted_imp[:top_k]
        sorted_names = sorted_names[:top_k]

        sns.barplot(x=sorted_imp, y=sorted_names, ax=ax, hue=sorted_names, palette="viridis", legend=False)
        ax.set_title(f"Feature Importance: {model_name}")
        ax.set_xlabel("Gini Importance")
        ax.set_xlim(0, 1.0)

    plt.tight_layout()

    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    out_file = os.path.join(PROJECT_ROOT, "results", "feature_importances.png")
    plt.savefig(out_file, dpi=200)
    print(f"Successfully generated Feature Importance plot: {out_file}")

if __name__ == "__main__":
    plot_feature_importances()