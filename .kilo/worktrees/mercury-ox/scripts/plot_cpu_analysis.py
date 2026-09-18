import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def create_cpu_analysis_plot():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="cpu_mflops_benchmark.csv", help="Input CSV path")
    parser.add_argument("--output", default="Fig13_CPU_MFlops_Analysis.png", help="Output PNG filename")
    args = parser.parse_args()

    csv_file = args.csv
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    os.makedirs("results", exist_ok=True)

    m_densities = [0.001, 0.01, 0.1, 1.0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    styles = {
        'LB-PM-BHash': {'marker': 'o', 'color': '
        'PM-BHash':    {'marker': 's', 'color': '
        'LB-PB-MSPA':  {'marker': 'D', 'color': '
        'PB-MSPA':     {'marker': 'h', 'color': '
        'LB-MSPA':     {'marker': 'v', 'color': '
    }

    for i, md in enumerate(m_densities):
        ax = axes[i]
        subset = df[df['MaskDensity'] == md]

        for algo, style in styles.items():
            algo_data = subset[subset['Kernel'] == algo]
            if not algo_data.empty:
                algo_data = algo_data.sort_values('XDensity')
                x_labels = algo_data['XDensity'].astype(str).tolist()

                ax.plot(x_labels, algo_data['MFlops'],
                        marker=style['marker'], color=style['color'],
                        label=algo, markersize=8, linewidth=2,
                        markeredgecolor='black', markeredgewidth=0.5)

        ax.set_title(f"Density of m = {md}")
        if i % 2 == 0: ax.set_ylabel("MFlops")
        if i >= 2: ax.set_xlabel("Density of x")
        if i == 3: ax.legend(loc='lower right', fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(f"results/{args.output}", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Success. Plot saved to results/{args.output}")

if __name__ == "__main__":
    create_cpu_analysis_plot()