import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Populate baseline and tuned benchmark datasets
    baseline_data = {
        'Concurrency': [1, 2, 3, 4, 5, 6, 7, 8],
        'Inferences/Second': [311.846, 591.095, 875.505, 1159.76, 1433.01, 1799.9, 2026.42, 3340.93],
        'Client Avg Latency': [3188, 3367, 3410, 3433, 3472, 3319, 3438, 2376],
        'p99 latency': [3995, 4011, 4449, 4738, 4826, 4423, 6088, 6209],
        'Server Queue': [2260, 2239, 2200, 2119, 2056, 2027, 1874, 554],
        'Server Compute Infer': [84, 90, 88, 89, 90, 80, 88, 91]
    }

    tuned_data = {
        'Concurrency': [1, 2, 3, 4, 5, 6, 7, 8],
        'Inferences/Second': [768.807, 1415.93, 1800.06, 2287.37, 2319.01, 2742.51, 4124.06, 4723.08],
        'Client Avg Latency': [1281, 1397, 1648, 1731, 2135, 2166, 1681, 1678],
        'p99 latency': [2364, 3152, 4313, 4474, 7377, 9642, 3984, 3724],
        'Server Queue': [453, 425, 414, 410, 456, 455, 373, 350],
        'Server Compute Infer': [76, 73, 82, 82, 93, 94, 79, 80]
    }

    df_base = pd.DataFrame(baseline_data)
    df_tuned = pd.DataFrame(tuned_data)

    # Save CSV files to disk
    df_base.to_csv('perf_report.csv', index=False)
    df_tuned.to_csv('perf_report_tuned.csv', index=False)
    print("[✓] Saved 'perf_report.csv' and 'perf_report_tuned.csv'.")

    # Generate comparison plots
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('CCVNN V14 Triton Optimization Analysis: Baseline (1000µs Queue) vs Tuned (250µs Dynamic Batching)', fontsize=14, fontweight='bold')

    # 1. Throughput
    axes[0].plot(df_base['Concurrency'], df_base['Inferences/Second'], 'o--', label='Baseline (1000 µs)', color='#1f77b4', linewidth=2)
    axes[0].plot(df_tuned['Concurrency'], df_tuned['Inferences/Second'], 'o-', label='Tuned (250 µs)', color='#2ca02c', linewidth=2)
    axes[0].set_title('Throughput Scaling (infer/sec)', fontweight='bold')
    axes[0].set_xlabel('Concurrency')
    axes[0].set_ylabel('Inferences / Second')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    # 2. Server Queue
    axes[1].plot(df_base['Concurrency'], df_base['Server Queue'], 'o--', label='Baseline', color='#d62728', linewidth=2)
    axes[1].plot(df_tuned['Concurrency'], df_tuned['Server Queue'], 'o-', label='Tuned', color='#ff7f0e', linewidth=2)
    axes[1].set_title('Server Queue Time (µs)', fontweight='bold')
    axes[1].set_xlabel('Concurrency')
    axes[1].set_ylabel('Queue Latency (µs)')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()

    # 3. P99 Tail Latency
    axes[2].plot(df_base['Concurrency'], df_base['p99 latency'], 'o--', label='Baseline', color='#9467bd', linewidth=2)
    axes[2].plot(df_tuned['Concurrency'], df_tuned['p99 latency'], 'o-', label='Tuned', color='#17becf', linewidth=2)
    axes[2].set_title('P99 Tail Latency (µs)', fontweight='bold')
    axes[2].set_xlabel('Concurrency')
    axes[2].set_ylabel('P99 Latency (µs)')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    
    os.makedirs("public", exist_ok=True)
    plt.savefig('triton_perf_comparison.png', dpi=300)
    plt.savefig('public/triton_perf_comparison.png', dpi=300)
    plt.close()
    print("[✓] Comparison graphs generated at 'triton_perf_comparison.png' and 'public/triton_perf_comparison.png'.")

if __name__ == "__main__":
    main()