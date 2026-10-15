import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    try:
        df_base = pd.read_csv('perf_report.csv')
        df_tuned = pd.read_csv('perf_report_tuned.csv')
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        print("Ensure 'perf_report.csv' and 'perf_report_tuned.csv' exist in the current directory.")
        sys.exit(1)

    # Normalize column names by trimming whitespace
    df_base.columns = df_base.columns.str.strip()
    df_tuned.columns = df_tuned.columns.str.strip()

    # Identify latency columns dynamically
    p99_col_base = [c for c in df_base.columns if 'p99' in c.lower()][0]
    p99_col_tuned = [c for c in df_tuned.columns if 'p99' in c.lower()][0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Triton Optimization Analysis: Baseline vs Tuned Queue Delay', fontsize=14, fontweight='bold')

    # Plot 1: Throughput Comparison
    axes[0].plot(df_base['Concurrency'], df_base['Inferences/Second'], 'o--', label='Baseline (1000 µs)', color='#1f77b4', linewidth=2)
    axes[0].plot(df_tuned['Concurrency'], df_tuned['Inferences/Second'], 'o-', label='Tuned (250 µs)', color='#2ca02c', linewidth=2)
    axes[0].set_title('Throughput Scaling (infer/sec)')
    axes[0].set_xlabel('Concurrency')
    axes[0].set_ylabel('Inferences / Second')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    # Plot 2: Server Queue Latency
    axes[1].plot(df_base['Concurrency'], df_base['Server Queue'], 'o--', label='Baseline', color='#d62728', linewidth=2)
    axes[1].plot(df_tuned['Concurrency'], df_tuned['Server Queue'], 'o-', label='Tuned', color='#ff7f0e', linewidth=2)
    axes[1].set_title('Server Queue Time (µs)')
    axes[1].set_xlabel('Concurrency')
    axes[1].set_ylabel('Queue Latency (µs)')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()

    # Plot 3: P99 Tail Latency
    axes[2].plot(df_base['Concurrency'], df_base[p99_col_base], 'o--', label='Baseline', color='#9467bd', linewidth=2)
    axes[2].plot(df_tuned['Concurrency'], df_tuned[p99_col_tuned], 'o-', label='Tuned', color='#17becf', linewidth=2)
    axes[2].set_title('P99 Tail Latency (µs)')
    axes[2].set_xlabel('Concurrency')
    axes[2].set_ylabel('P99 Latency (µs)')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    output_img = 'triton_perf_comparison.png'
    plt.savefig(output_img, dpi=300)
    print(f"Comparison graph generated and saved to '{output_img}'.")

if __name__ == '__main__':
    main()