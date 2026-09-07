# File: scripts/plot_benchmark.py
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_benchmark_plot(json_path="benchmark_results.json", output_image="docs/glass_to_glass_latency.png"):
    if not os.path.exists(json_path):
        print(f"[!] File not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    summary = data["summary_ms"]
    raw_totals_ms = np.array(data["frame_totals_us"]) / 1000.0

    stages = [
        "1. Acquisition",
        "2. Extraction",
        "3. Inference",
        "4. Hysteresis",
        "5. GPIO Output",
        "TOTAL Glass-to-Glass"
    ]
    keys = [
        "frame_acquisition",
        "spatial_extraction",
        "ccvnn_inference",
        "hysteresis_filter",
        "gpio_relay",
        "total_glass_to_glass"
    ]

    p50 = [summary[k]["p50"] for k in keys]
    p95 = [summary[k]["p95"] for k in keys]
    p99 = [summary[k]["p99"] for k in keys]

    os.makedirs(os.path.dirname(output_image), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Stage Percentile Breakdown
    x = np.arange(len(stages))
    width = 0.25

    ax1.bar(x - width, p50, width, label="p50 Latency", color="#2b5c8f")
    ax1.bar(x, p95, width, label="p95 Latency", color="#d95f02")
    ax1.bar(x + width, p99, width, label="p99 Latency", color="#7570b3")

    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Stage-by-Stage Latency Percentiles")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=25, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.legend()

    # Panel 2: Frame-by-Frame Latency Distribution
    ax2.plot(raw_totals_ms, alpha=0.75, color="#1b9e77", linewidth=1, label="Total Glass-to-Glass")
    ax2.axhline(y=summary["total_glass_to_glass"]["p50"], color="#2b5c8f", linestyle="--", label=f"p50 ({summary['total_glass_to_glass']['p50']:.2f} ms)")
    ax2.axhline(y=summary["total_glass_to_glass"]["p99"], color="#7570b3", linestyle=":", label=f"p99 ({summary['total_glass_to_glass']['p99']:.2f} ms)")

    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Frame Telemetry & Stability")
    ax2.grid(linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"[✓] Benchmark visualization saved to: {output_image}")

if __name__ == "__main__":
    generate_benchmark_plot()
