import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_side_by_side():
    files = sorted(glob.glob("benchmark_batch_*.csv"))
    if not files:
        print("No benchmark CSV files found. Please run ./run_sweep.sh first.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    plotted_any = False

    for file_path in files:
        batch_size = file_path.split("_")[-1].replace(".csv", "")
        
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            
            # Map correctly to 'inferences/second' and exact latency column names
            throughput_col = next((c for c in df.columns if 'inferences/second' in c or 'throughput' in c), None)
            p50_col = next((c for c in df.columns if 'p50 latency' in c), None)
            p99_col = next((c for c in df.columns if 'p99 latency' in c), None)

            if not throughput_col or not p50_col or not p99_col:
                print(f"Skipping {file_path}: Columns found -> {list(df.columns)}")
                continue

            throughput = df[throughput_col]
            p50_latency = df[p50_col] / 1000.0  # Convert usec to ms
            p99_latency = df[p99_col] / 1000.0  # Convert usec to ms

            ax1.plot(throughput, p50_latency, marker='o', linestyle='-', label=f"Batch {batch_size}")
            ax2.plot(throughput, p99_latency, marker='s', linestyle='--', label=f"Batch {batch_size}")
            plotted_any = True

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not plotted_any:
        print("Error: No files were successfully parsed.")
        return

    ax1.set_title("P50 Latency vs Throughput", fontsize=12)
    ax1.set_xlabel("Throughput (infer/sec)", fontsize=10)
    ax1.set_ylabel("P50 Latency (ms)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    ax2.set_title("P99 Tail Latency vs Throughput", fontsize=12)
    ax2.set_xlabel("Throughput (infer/sec)", fontsize=10)
    ax2.set_ylabel("P99 Latency (ms)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.suptitle("Triton Performance Benchmark: P50 vs P99 Latency Profiles", fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_image = "latency_p50_p99_comparison.png"
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Side-by-side comparison plot successfully saved to {output_image}")

if __name__ == "__main__":
    plot_side_by_side()
