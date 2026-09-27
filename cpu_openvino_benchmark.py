import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_perf_csv(filepath):
    """
    Parses Triton perf_analyzer CSV file and standardizes key metrics.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Normalize column names for flexible matching
    column_mapping = {}
    for col in df.columns:
        col_clean = col.strip().lower()
        if "concurrency" in col_clean:
            column_mapping[col] = "concurrency"
        elif "inferences/second" in col_clean or "throughput" in col_clean:
            column_mapping[col] = "throughput"
        elif "p99" in col_clean:
            column_mapping[col] = "p99_latency"

    df = df.rename(columns=column_mapping)

    # Convert latency from microseconds (us) to milliseconds (ms) if needed
    if "p99_latency" in df.columns and df["p99_latency"].mean() > 500:
        df["p99_latency_ms"] = df["p99_latency"] / 1000.0
    else:
        df["p99_latency_ms"] = df["p99_latency"]

    return df


def plot_side_by_side(onnx_csv, openvino_csv, output_image="benchmark_comparison.png"):
    # Load dataset
    df_onnx = parse_perf_csv(onnx_csv)
    df_ov = parse_perf_csv(openvino_csv)

    # Set up styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Triton Benchmark: ONNX Runtime vs OpenVINO IR (4 CPU Cores)", fontsize=14, fontweight="bold")

    # Colors
    color_onnx = "#1f77b4"  # Blue
    color_ov = "#2ca02c"    # Green

    # Plot 1: Concurrency vs Throughput (FPS)
    axes[0].plot(df_onnx["concurrency"], df_onnx["throughput"], marker="o", linewidth=2, color=color_onnx, label="ONNX FP32")
    axes[0].plot(df_ov["concurrency"], df_ov["throughput"], marker="s", linewidth=2, color=color_ov, label="OpenVINO INT8/FP16")
    axes[0].set_title("Throughput vs Concurrency")
    axes[0].set_xlabel("Concurrency Level")
    axes[0].set_ylabel("Throughput (Inferences / sec)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Concurrency vs P99 Latency (ms)
    axes[1].plot(df_onnx["concurrency"], df_onnx["p99_latency_ms"], marker="o", linewidth=2, color=color_onnx, label="ONNX FP32")
    axes[1].plot(df_ov["concurrency"], df_ov["p99_latency_ms"], marker="s", linewidth=2, color=color_ov, label="OpenVINO INT8/FP16")
    axes[1].set_title("P99 Latency vs Concurrency")
    axes[1].set_xlabel("Concurrency Level")
    axes[1].set_ylabel("P99 Latency (ms)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Plot 3: Throughput vs P99 Latency Curve (Efficiency Frontier)
    axes[2].plot(df_onnx["throughput"], df_onnx["p99_latency_ms"], marker="o", linewidth=2, color=color_onnx, label="ONNX FP32")
    axes[2].plot(df_ov["throughput"], df_ov["p99_latency_ms"], marker="s", linewidth=2, color=color_ov, label="OpenVINO INT8/FP16")
    axes[2].set_title("Throughput vs P99 Latency (Trade-off Curve)")
    axes[2].set_xlabel("Throughput (Inferences / sec)")
    axes[2].set_ylabel("P99 Latency (ms)")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"[+] Benchmark comparison plot saved to: {output_image}")
    plt.show()


if __name__ == "__main__":
    onnx_file = "onnx_benchmark.csv"
    openvino_file = "cpu_openvino_benchmark.csv"

    plot_side_by_side(onnx_file, openvino_file)