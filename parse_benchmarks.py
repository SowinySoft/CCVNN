import glob
import os
import pandas as pd

def parse_results():
    files = glob.glob("benchmark_batch_*.csv")
    if not files:
        print("No benchmark CSV files found. Run ./run_sweep.sh first.")
        return

    summary_data = []

    for file_path in sorted(files):
        # Extract batch size from filename
        batch_size = file_path.split("_")[-1].replace(".csv", "")
        
        try:
            df = pd.read_csv(file_path)
            
            # perf_analyzer columns usually include: 
            # 'Batch', 'Concurrencies', 'Throughput', 'p50 latency', 'p90 latency', 'p99 latency'
            # Let's handle variations in column naming defensively:
            throughput_col = next((c for c in df.columns if 'throughput' in c.lower()), None)
            latency_col = next((c for c in df.columns if 'latency' in c.lower() and 'avg' in c.lower()), None)
            p99_col = next((c for c in df.columns if '99' in c), None)

            if not throughput_col:
                # Fallback to standard perf_analyzer column layout
                # Typically: [client_response_total, infer_per_sec, send_time, ... , latency, ...]
                # Let's inspect rows or use positional fallbacks if columns differ
                pass

            # Read the last row or aggregate rows across concurrencies
            # For a quick summary, we can take the peak throughput or average across concurrency
            avg_throughput = df[throughput_col].mean() if throughput_col else df.iloc[:, 1].mean()
            
            summary_data.append({
                "Batch Size": batch_size,
                "Avg Throughput (infer/sec)": round(avg_throughput, 2),
            })
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\n=== Triton Benchmark Summary Table ===")
        print(summary_df.to_markdown(index=False))

if __name__ == "__main__":
    parse_results()
