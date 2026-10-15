import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Discover all benchmark CSV files
csv_files = sorted(glob.glob("benchmark_batch_*.csv"))

if not csv_files:
    print("No benchmark_batch_*.csv files found in current directory.")
    exit(1)

batch_data = []

def find_column(df, patterns):
    """Find a column matching any of the given patterns."""
    for pattern in patterns:
        for col in df.columns:
            if re.search(pattern, col, re.IGNORECASE):
                return col
    return None

for file_path in csv_files:
    match = re.search(r"benchmark_batch_(\d+)\.csv", file_path)
    batch_size = int(match.group(1)) if match else file_path

    df = pd.read_csv(file_path)

    # Detect throughput and timing columns
    throughput_col = find_column(df, [r"Inferences/Second", r"Throughput"])
    queue_col = find_column(df, [r"Server Queue", r"Queue Time", r"queue"])
    compute_infer_col = find_column(df, [r"Server Compute Infer", r"Compute Infer", r"compute_infer"])
    compute_input_col = find_column(df, [r"Server Compute Input", r"Compute Input"])
    compute_output_col = find_column(df, [r"Server Compute Output", r"Compute Output"])

    if not queue_col or not compute_infer_col:
        print(f"Skipping {file_path}: couldn't identify queue/compute columns.")
        continue

    # Select peak throughput row to compare latency under maximum load
    peak_row = df.loc[df[throughput_col].idxmax()] if throughput_col else df.iloc[-1]

    queue_time = peak_row[queue_col]
    compute_time = peak_row[compute_infer_col]
    
    # Add input/output handling time if present in Triton CSV
    if compute_input_col and pd.notna(peak_row[compute_input_col]):
        compute_time += peak_row[compute_input_col]
    if compute_output_col and pd.notna(peak_row[compute_output_col]):
        compute_time += peak_row[compute_output_col]

    # Normalize to milliseconds if values are in microseconds (usec/us)
    unit_check_str = f"{queue_col} {compute_infer_col}".lower()
    if "usec" in unit_check_str or "us" in unit_check_str:
        queue_time /= 1000.0
        compute_time /= 1000.0

    batch_data.append({
        "batch_size": f"Batch {batch_size}",
        "queue_ms": queue_time,
        "compute_ms": compute_time,
        "total_ms": queue_time + compute_time
    })

# Convert summary data to DataFrame
summary_df = pd.DataFrame(batch_data)

# Create Stacked Bar Chart
plt.figure(figsize=(9, 6))
bars_compute = plt.bar(
    summary_df["batch_size"], 
    summary_df["compute_ms"], 
    label="Compute Time (ms)", 
    color="#2b5c8f"
)
bars_queue = plt.bar(
    summary_df["batch_size"], 
    summary_df["queue_ms"], 
    bottom=summary_df["compute_ms"], 
    label="Server Queue Time (ms)", 
    color="#d95f02"
)

# Add data value labels inside bars
for b_comp, b_q, total in zip(bars_compute, bars_queue, summary_df["total_ms"]):
    h_comp = b_comp.get_height()
    h_q = b_q.get_height()
    
    if h_comp > 0.1:
        plt.text(b_comp.get_x() + b_comp.get_width()/2., h_comp / 2,
                 f"{h_comp:.2f} ms", ha='center', va='center', color='white', fontweight='bold')
    if h_q > 0.1:
        plt.text(b_q.get_x() + b_q.get_width()/2., h_comp + (h_q / 2),
                 f"{h_q:.2f} ms", ha='center', va='center', color='white', fontweight='bold')

plt.ylabel("Latency Breakdown (ms)", fontsize=12)
plt.title("Triton Latency Breakdown: Queue Time vs. Compute Time (at Peak Load)", fontsize=13)
plt.legend(loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

output_file = "latency_breakdown_queue_vs_compute.png"
plt.savefig(output_file, dpi=300)
print(f"Successfully generated plot: {output_file}")