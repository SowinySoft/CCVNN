import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_triton_perf(csv_file, output_image="perf_curve.png"):
    # Load perf_analyzer CSV output
    df = pd.read_csv(csv_file)

    # Standardize column header naming variations across Triton SDK releases
    df.columns = [c.strip() for c in df.columns]

    concurrency_col = [c for c in df.columns if "Concurrency" in c][0]
    throughput_col = [c for c in df.columns if "Inferences/Second" in c or "Throughput" in c][0]
    latency_col = [c for c in df.columns if "p95" in c or "Avg latency" in c or "Latency" in c][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Latency vs Throughput (Efficiency Curve)
    ax1.plot(df[throughput_col], df[latency_col], marker='o', linewidth=2, color='#2b5c8f')
    for i, txt in enumerate(df[concurrency_col]):
        ax1.annotate(f"C={txt}", (df[throughput_col].iloc[i], df[latency_col].iloc[i]),
                     textcoords="offset points", xytext=(5,5), ha='left')
    ax1.set_title("Latency vs. Throughput Curve")
    ax1.set_xlabel("Throughput (Inferences / Second)")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Concurrency vs Metrics Scaling
    ax2.plot(df[concurrency_col], df[throughput_col], marker='s', color='#2e7d32', label="Throughput (infer/sec)")
    ax2_lat = ax2.twinx()
    ax2_lat.plot(df[concurrency_col], df[latency_col], marker='^', color='#c62828', label="Latency (ms)")

    ax2.set_title("Concurrency vs. Throughput & Latency")
    ax2.set_xlabel("Concurrency Level")
    ax2.set_ylabel("Throughput (infer/sec)", color='#2e7d32')
    ax2_lat.set_ylabel("Latency (ms)", color='#c62828')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"[SUCCESS] Saved performance benchmark visualization to {output_image}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ccvnn_perf_results.csv"
    plot_triton_perf(file_path)