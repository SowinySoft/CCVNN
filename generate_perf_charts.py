#!/usr/bin/env python3
import csv
from collections import defaultdict
import glob
import os
import re

# Use non-interactive backend for server/headless environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_float(val, default=0.0):
    """Safely convert value or string to float, stripping non-numeric characters if needed."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))
    return float(match.group(0)) if match else default


def find_column_val(row, candidate_patterns):
    """Search DictReader row for column names matching candidate regex patterns."""
    normalized_row = {
        re.sub(r"\s+", " ", k.strip().lower()): v
        for k, v in row.items()
        if k
    }

    for pattern in candidate_patterns:
        pattern_clean = pattern.lower()
        if pattern_clean in normalized_row:
            return normalized_row[pattern_clean]
        for norm_key, val in normalized_row.items():
            if re.search(pattern_clean, norm_key):
                return val
    return None


def parse_perf_csv_files():
    """Find and parse all perf_*.csv files in the current directory."""
    csv_files = glob.glob("perf_*.csv")
    if not csv_files:
        print("[-] No 'perf_*.csv' files found in current directory.")
        return None

    # Structure: data[model_name][batch_size][concurrency] = metrics_dict
    data = defaultdict(lambda: defaultdict(dict))

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        match = re.match(r"perf_(.+)_batch_(\d+)\.csv", filename)
        if not match:
            continue

        model_name, batch_size = match.group(1), int(match.group(2))

        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_conc = find_column_val(
                    row, ["concurrency", "conc", "client concurrency"]
                )
                raw_tp = find_column_val(
                    row, ["inferences/second", "throughput", "inferences_per_second"]
                )
                raw_avg = find_column_val(
                    row, ["avg latency", "client latency", "avg_latency", "p50 latency"]
                )
                raw_p95 = find_column_val(
                    row, ["p95 latency", "p95_latency", "95th percentile"]
                )
                raw_queue = find_column_val(
                    row, ["server queue", "queue latency", "server_queue"]
                )
                raw_compute = find_column_val(
                    row, ["server compute infer", "compute latency", "server_compute"]
                )

                conc = int(safe_float(raw_conc, default=1))
                throughput = safe_float(raw_tp)

                # Microseconds to milliseconds conversion
                avg_lat = safe_float(raw_avg) / 1000.0
                p95_lat = safe_float(raw_p95) / 1000.0
                queue_lat = safe_float(raw_queue) / 1000.0
                compute_lat = safe_float(raw_compute) / 1000.0

                # Fallback: calculate total server latency if client avg latency is unparsed/0
                if avg_lat == 0.0 and (queue_lat > 0 or compute_lat > 0):
                    avg_lat = queue_lat + compute_lat

                data[model_name][batch_size][conc] = {
                    "throughput": throughput,
                    "sample_throughput": throughput * batch_size,
                    "avg_lat": avg_lat,
                    "p95_lat": p95_lat,
                    "queue_lat": queue_lat,
                    "compute_lat": compute_lat,
                }

    return data


def plot_performance_dashboard(data, output_filepath="triton_performance_charts.png"):
    """Generate a 4-panel performance dashboard PNG."""
    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    fig.suptitle(
        "Triton Inference Server Performance Analysis [ -1, 4, 64, 64 ]",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    models = sorted(data.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    model_color_map = {model: colors[i % len(colors)] for i, model in enumerate(models)}

    # -------------------------------------------------------------
    # Plot 1: Sample Throughput (Samples/sec) vs. Batch Size
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    for model in models:
        batch_sizes = sorted(data[model].keys())
        sample_throughputs = []
        for b in batch_sizes:
            c1 = min(data[model][b].keys())
            sample_throughputs.append(data[model][b][c1]["sample_throughput"])

        color = model_color_map[model]
        ax1.plot(
            batch_sizes,
            sample_throughputs,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=f"{model}",
            color=color,
        )
        for b, val in zip(batch_sizes, sample_throughputs):
            ax1.annotate(
                f"{val:.1f}",
                (b, val),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    ax1.set_title("Sample Throughput vs. Request Batch Size (Concurrency = 1)")
    ax1.set_xlabel("Batch Size (B)")
    ax1.set_ylabel("Throughput (Samples / sec)")
    ax1.set_xticks([1, 4, 8, 16, 32])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # -------------------------------------------------------------
    # Plot 2: Average Latency vs. Batch Size
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    for model in models:
        batch_sizes = sorted(data[model].keys())
        latencies = []
        for b in batch_sizes:
            c1 = min(data[model][b].keys())
            latencies.append(data[model][b][c1]["avg_lat"])

        color = model_color_map[model]
        ax2.plot(
            batch_sizes,
            latencies,
            marker="s",
            linewidth=2.5,
            markersize=8,
            label=f"{model}",
            color=color,
        )
        for b, val in zip(batch_sizes, latencies):
            ax2.annotate(
                f"{val:.2f}ms",
                (b, val),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    ax2.set_title("Average Latency vs. Request Batch Size (Concurrency = 1)")
    ax2.set_xlabel("Batch Size (B)")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_xticks([1, 4, 8, 16, 32])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    # -------------------------------------------------------------
    # Plots 3 & 4: Server Time Breakdown (Queue vs. Compute)
    # -------------------------------------------------------------
    plot_indices = [(1, 0), (1, 1)]
    for idx, model in enumerate(models[:2]):
        ax = axes[plot_indices[idx]]
        batch_sizes = sorted(data[model].keys())

        compute_times = []
        queue_times = []

        for b in batch_sizes:
            c1 = min(data[model][b].keys())
            compute_times.append(data[model][b][c1]["compute_lat"])
            queue_times.append(data[model][b][c1]["queue_lat"])

        bar_width = 0.4
        x_indices = range(len(batch_sizes))

        ax.bar(
            x_indices,
            compute_times,
            width=bar_width,
            label="Compute Infer Time",
            color="#2ca02c",
            alpha=0.85,
        )
        ax.bar(
            x_indices,
            queue_times,
            width=bar_width,
            bottom=compute_times,
            label="Queue Delay",
            color="#d62728",
            alpha=0.85,
        )

        # Annotate stacked bar totals
        for i, (comp, q) in enumerate(zip(compute_times, queue_times)):
            total = comp + q
            ax.text(
                i,
                total + (total * 0.02),
                f"{total:.2f}ms",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(f"Latency Breakdown: {model}")
        ax.set_xlabel("Batch Size (B)")
        ax.set_ylabel("Time (ms)")
        ax.set_xticks(x_indices)
        ax.set_xticklabels([str(b) for b in batch_sizes])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filepath, dpi=300)
    print(f"[+] High-resolution charts saved to '{output_filepath}'")


def main():
    data = parse_perf_csv_files()
    if data:
        plot_performance_dashboard(data)


if __name__ == "__main__":
    main()