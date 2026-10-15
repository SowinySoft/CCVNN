#!/usr/bin/env python3
import argparse
import io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_benchmark_csv(csv_path: str, output_path: str):
    """Loads benchmark CSV output and generates publication-grade Histogram and CDF plots."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found at: {csv_path}")
        return

    # Read file and locate data table start (skipping metadata key-value lines)
    with open(csv_path, "r") as f:
        lines = f.readlines()

    data_start_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith("request_id"):
            data_start_idx = idx
            break

    data_str = "".join(lines[data_start_idx:])
    df = pd.read_csv(io.StringIO(data_str))

    # Clean & filter data
    if "status" in df.columns:
        df = df[df["status"] == "SUCCESS"]

    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    latencies = df["latency_ms"].dropna().values

    if len(latencies) == 0:
        print("[ERROR] No valid latency data found in CSV.")
        return

    # Compute key statistics
    mean_lat = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    # Style configuration
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --- Subplot A: Latency Distribution Histogram ---
    counts, bins, patches = ax1.hist(
        latencies, bins=30, density=True, color="#2a9d8f", edgecolor="#1d3557", alpha=0.75
    )
    
    ax1.axvline(mean_lat, color="#005f73", linestyle="--", linewidth=1.8, label=f"Mean: {mean_lat:.2f} ms")
    ax1.axvline(p95, color="#e9c46a", linestyle="--", linewidth=1.8, label=f"p95: {p95:.2f} ms")
    ax1.axvline(p99, color="#e76f51", linestyle="-.", linewidth=2.0, label=f"p99: {p99:.2f} ms")

    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("A: Latency Distribution Histogram")
    ax1.legend(loc="upper right")

    # --- Subplot B: Empirical CDF ---
    sorted_lat = np.sort(latencies)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)

    ax2.plot(sorted_lat, cdf, color="#005f73", linewidth=2.2, label="Empirical CDF")

    # Annotate Percentile Markers
    percentile_targets = [(50, p50, "#2a9d8f"), (95, p95, "#e9c46a"), (99, p99, "#e76f51")]
    for p, val, col in percentile_targets:
        ax2.axvline(val, color=col, linestyle=":", alpha=0.8)
        ax2.axhline(p / 100.0, color=col, linestyle=":", alpha=0.8)
        ax2.plot(val, p / 100.0, "o", color=col, label=f"p{p}: {val:.2f} ms")

    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("B: Cumulative Distribution Function (CDF)")
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(loc="lower right")

    plt.suptitle(f"Async gRPC Request Latency Profile ({len(latencies)} Requests)", y=1.01)
    plt.tight_layout()

    # Save outputs
    abs_out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    plt.savefig(abs_out, dpi=300, bbox_inches="tight")
    print(f"[SUCCESS] Latency CDF & Histogram chart saved to: {abs_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Latency CDF and Histogram plot from benchmark CSV file."
    )
    parser.add_argument("-i", "--input", type=str, default="benchmark_results.csv",
                        help="Path to input benchmark CSV file.")
    parser.add_argument("-o", "--output", type=str, default="latency_cdf_histogram.png",
                        help="Path to save rendered plot (e.g. PNG or SVG).")

    args = parser.parse_args()
    plot_benchmark_csv(args.input, args.output)


if __name__ == "__main__":
    main()