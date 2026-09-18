import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCRIPT_DIR, "realworld_results.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")

def plot_results():
    if not os.path.exists(RESULTS_CSV):
        print(f"Results file {RESULTS_CSV} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Dataset", y="Time_ms", hue="Variant")
    plt.yscale("log")
    plt.title("Realworld Graph Performance: Original vs Improved")
    plt.ylabel("Execution Time (ms) - Log Scale")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "realworld_time_comparison.png"))

    baseline = df[df["Variant"] == "ORIGINAL-BASELINE"].set_index("Dataset")["Time_ms"]
    df["Speedup"] = df.apply(lambda row: baseline.get(row["Dataset"], row["Time_ms"]) / row["Time_ms"], axis=1)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df[df["Variant"] != "ORIGINAL-BASELINE"], x="Dataset", y="Speedup", hue="Variant")
    plt.axhline(1.0, color='red', linestyle='--')
    plt.title("Speedup Relative to Paper Baseline")
    plt.ylabel("Speedup (x)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "realworld_speedup.png"))

    print(f"\n✓ Comparison plots generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    plot_results()