# plot_benchmark.py
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_benchmark_data(json_file: str, output_image: str):
    with open(json_file, "r") as f:
        data = json.load(f)

    runs = data["runs"]
    concurrencies = [r["concurrency"] for r in runs]
    throughputs = [r["throughput_fps"] for r in runs]
    p50 = [r["p50_latency_ms"] for r in runs]
    p90 = [r["p90_latency_ms"] for r in runs]
    p99 = [r["p99_latency_ms"] for r in runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"SLA Benchmark Performance: {data['model_name']}", fontsize=14, fontweight="bold")

    # 1. Throughput vs Concurrency
    ax1.plot(concurrencies, throughputs, marker="o", color="#1f77b4", linewidth=2.5, label="Throughput")
    ax1.set_title("Throughput vs. Concurrency", fontsize=12)
    ax1.set_xlabel("Concurrent Worker Tasks", fontsize=10)
    ax1.set_ylabel("Inferences / Second (FPS)", fontsize=10)
    ax1.set_xticks(concurrencies)
    ax1.grid(True, linestyle="--", alpha=0.6)
    for i, txt in enumerate(throughputs):
        ax1.annotate(f"{txt:.0f}", (concurrencies[i], throughputs[i]), textcoords="offset points", xytext=(0, 8), ha="center")

    # 2. Latency Percentiles vs Concurrency
    ax2.plot(concurrencies, p50, marker="s", color="#2ca02c", linewidth=2, label="P50 (Median)")
    ax2.plot(concurrencies, p90, marker="^", color="#ff7f0e", linewidth=2, label="P90")
    ax2.plot(concurrencies, p99, marker="d", color="#d62728", linewidth=2, label="P99")
    ax2.set_title("Latency Percentiles vs. Concurrency", fontsize=12)
    ax2.set_xlabel("Concurrent Worker Tasks", fontsize=10)
    ax2.set_ylabel("Latency (ms)", fontsize=10)
    ax2.set_xticks(concurrencies)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Successfully generated benchmark visualization: '{output_image}'")


def plot_single_run_histogram(json_file: str, concurrency_target: int, output_image: str):
    with open(json_file, "r") as f:
        data = json.load(f)

    target_run = None
    for r in data["runs"]:
        if r["concurrency"] == concurrency_target:
            target_run = r
            break

    if not target_run or "latencies_raw" not in target_run:
        print(f"No raw latency data found for concurrency={concurrency_target}")
        return

    latencies = target_run["latencies_raw"]
    plt.figure(figsize=(8, 5))
    plt.hist(latencies, bins=40, color="#4c72b0", edgecolor="black", alpha=0.7)

    plt.axvline(target_run["p50_latency_ms"], color="#2ca02c", linestyle="--", linewidth=2, label=f"P50: {target_run['p50_latency_ms']:.2f}ms")
    plt.axvline(target_run["p90_latency_ms"], color="#ff7f0e", linestyle="--", linewidth=2, label=f"P90: {target_run['p90_latency_ms']:.2f}ms")
    plt.axvline(target_run["p99_latency_ms"], color="#d62728", linestyle="--", linewidth=2, label=f"P99: {target_run['p99_latency_ms']:.2f}ms")

    plt.title(f"Latency Distribution (Concurrency = {concurrency_target})", fontsize=12, fontweight="bold")
    plt.xlabel("Latency (ms)", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Successfully generated distribution plot: '{output_image}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Triton Benchmark Metrics")
    parser.add_argument("--json", type=str, default="benchmark_results.json", help="Input JSON results file")
    parser.add_argument("--out-summary", type=str, default="benchmark_summary.png", help="Summary plot output image")
    parser.add_argument("--out-dist", type=str, default="latency_distribution.png", help="Distribution plot output image")
    parser.add_argument("--target-concurrency", type=int, default=8, help="Target concurrency for latency histogram")
    args = parser.parse_args()

    plot_benchmark_data(args.json, args.out_summary)
    plot_single_run_histogram(args.json, args.target_concurrency, args.out_dist)