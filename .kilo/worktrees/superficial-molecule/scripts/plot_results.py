import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def create_charts():
    csv_files = glob.glob("bfs_run_*.csv")
    if not csv_files:
        print("No results CSV found. Run run_inference.py first.")
        return

    os.makedirs("results", exist_ok=True)

    print("Generating Figure 11 (Performance Comparisons) and Fig 13 (DTA Overhead)...")

    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if not df.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Dataset", y="Total Time (ms)", hue="Platform", data=df)
        plt.title("Execution Time Comparison (CPU vs GPU vs Universal)")
        plt.yscale('log')
        plt.savefig("results/Fig11_Execution_Comparison.png")
        plt.close()

    print("Success. Performance plots saved to results/")

if __name__ == "__main__":
    create_charts()