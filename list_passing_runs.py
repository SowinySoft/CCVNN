import glob
import pandas as pd

def list_passing():
    files = sorted(glob.glob("benchmark_batch_*.csv"))
    passing_runs = []

    for file_path in files:
        batch_size = file_path.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        
        throughput_col = next((c for c in df.columns if 'inferences/second' in c or 'throughput' in c), None)
        p99_col = next((c for c in df.columns if 'p99 latency' in c), None)

        if not throughput_col or not p99_col:
            continue

        # Adjusted threshold to 15,000 usec (15 ms) to match target SLA
        passing = df[df[p99_col] <= 15000]
        for _, row in passing.iterrows():
            passing_runs.append({
                'batch_size': batch_size,
                'concurrency': int(row['concurrency']),
                'throughput': row[throughput_col],
                'p99_ms': row[p99_col] / 1000.0
            })

    print(f"\n=== Passing Runs (P99 <= 15.0 ms) [Total: {len(passing_runs)}] ===")
    for run in passing_runs:
        print(f"Batch Size: {run['batch_size']} | Concurrency: {run['concurrency']} | Throughput: {run['throughput']:.2f} infer/sec | P99: {run['p99_ms']:.2f} ms")

if __name__ == "__main__":
    list_passing()
