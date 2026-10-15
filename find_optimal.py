import os
import glob
import pandas as pd

def find_optimal_config():
    files = sorted(glob.glob("benchmark_batch_*.csv"))
    if not files:
        print("[!] No benchmark CSV files matching 'benchmark_batch_*.csv' found.")
        return

    max_p99_ms = float(os.getenv("MAX_P99_MS", "15.0"))
    max_p99_usec = max_p99_ms * 1000.0

    best_config = None
    max_throughput = -1.0

    for file_path in files:
        batch_size = file_path.split("_")[-1].replace(".csv", "")
        
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            
            throughput_col = next((c for c in df.columns if 'inferences/second' in c or 'throughput' in c), None)
            p99_col = next((c for c in df.columns if 'p99 latency' in c or 'p99' in c), None)
            concurrency_col = next((c for c in df.columns if 'concurrency' in c), None)

            if not throughput_col or not p99_col:
                print(f"[!] Skipping {file_path}: missing required throughput/p99 columns.")
                continue

            # Filter configurations where p99 latency meets SLA
            valid_df = df[df[p99_col] < max_p99_usec]

            if not valid_df.empty:
                local_best = valid_df.loc[valid_df[throughput_col].idxmax()]
                if local_best[throughput_col] > max_throughput:
                    max_throughput = local_best[throughput_col]
                    concurrency_val = int(local_best[concurrency_col]) if concurrency_col else -1
                    best_config = {
                        'batch_size': batch_size,
                        'concurrency': concurrency_val,
                        'throughput': float(local_best[throughput_col]),
                        'p99_latency_ms': float(local_best[p99_col]) / 1000.0
                    }
        except Exception as e:
            print(f"[!] Error processing {file_path}: {e}")

    if best_config:
        print(f"\n=== Optimal CCVNN V14 Configuration Found (SLA < {max_p99_ms} ms) ===")
        print(f"Batch Size:     {best_config['batch_size']}")
        print(f"Concurrency:    {best_config['concurrency']}")
        print(f"Throughput:     {best_config['throughput']:.2f} infer/sec")
        print(f"P99 Latency:    {best_config['p99_latency_ms']:.2f} ms")
        print("=================================================================")
    else:
        print(f"[!] No configuration met the P99 latency threshold of {max_p99_ms} ms.")

if __name__ == "__main__":
    find_optimal_config()