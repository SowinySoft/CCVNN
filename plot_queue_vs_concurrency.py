import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# Discover benchmark CSV files
csv_files = sorted(glob.glob("benchmark_batch_*.csv"))

if not csv_files:
    print("No benchmark_batch_*.csv files found in current directory.")
    exit(1)

def find_column(df, patterns):
    """Find a column matching any of the given regex patterns."""
    for pattern in patterns:
        for col in df.columns:
            if re.search(pattern, col, re.IGNORECASE):
                return col
    return None

plt.figure(figsize=(10, 6))

for file_path in csv_files:
    match = re.search(r"benchmark_batch_(\d+)\.csv", file_path)
    batch_size = match.group(1) if match else file_path

    df = pd.read_csv(file_path)

    # Detect concurrency and queue time columns
    concurrency_col = find_column(df, [r"Concurrency", r"Concurrency Level"])
    queue_col = find_column(df, [r"Server Queue", r"Queue Time", r"queue"])

    if not concurrency_col or not queue_col:
        print(f"Skipping {file_path}: missing concurrency or queue time column.")
        continue

    # Convert queue time from microseconds to milliseconds if needed
    queue_values = df[queue_col].copy()
    if "usec" in queue_col.lower() or "us" in queue_col.lower():
        queue_values /= 1000.0

    # Sort data points by concurrency level
    df_sorted = df.assign(queue_ms=queue_values).sort_values(by=concurrency_col)

    plt.plot(
        df_sorted[concurrency_col],
        df_sorted["queue_ms"],
        marker="o",
        linewidth=2,
        label=f"Batch {batch_size}"
    )

plt.xlabel("Concurrency Level", fontsize=12)
plt.ylabel("Server Queue Time (ms)", fontsize=12)
plt.title("Triton Server Queue Time Scaling vs. Concurrency Level", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Batch Size")
plt.tight_layout()

output_file = "queue_vs_concurrency.png"
plt.savefig(output_file, dpi=300)
print(f"Successfully generated plot: {output_file}")