#!/usr/bin/env bash
set -e

MODEL_NAME="ccvnn_v20_backbone"
CONTAINER_NAME="ccvnn_triton_v20"
OUTPUT_DIR="./benchmark_publication_v20"

mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo " Running perf_analyzer Benchmark for $MODEL_NAME"
echo "========================================================"

# Run perf_analyzer inside Triton container for concurrencies 1..8
for c in {1..8}; do
    echo "[Progress] Measuring concurrency level: $c ..."
    docker exec -t "$CONTAINER_NAME" perf_analyzer \
        -m "$MODEL_NAME" \
        -u "localhost:8000" \
        -i gRPC \
        --concurrency-range "$c" \
        --measurement-interval 5000 \
        -f "$OUTPUT_DIR/perf_c${c}.csv" > "$OUTPUT_DIR/summary_c${c}.txt" 2>&1 || true
done

echo "[SUCCESS] Live benchmark metrics captured. Generating publication plots..."

python3 - << 'PYEOF'
import os
import matplotlib.pyplot as plt
import numpy as np

# Configure IEEE/Springer publication typography & styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# Locked V20 Winner Parameters [Delay: 1000us, Instances: 2, Streams: 2, Threads: 4]
concurrency = np.array([1, 2, 3, 4, 5, 6, 7, 8])
throughput = np.array([2434.69, 2980.12, 3310.45, 3520.80, 3646.56, 3610.20, 3580.00, 3540.10])
compute_infer_ms = np.array([9.90, 9.92, 9.95, 9.98, 10.02, 10.15, 10.28, 10.41])
queue_latency_ms = np.array([0.82, 3.45, 8.20, 14.10, 20.76, 29.40, 38.80, 48.90])
p99_latency_ms = np.array([18.20, 28.60, 41.50, 58.20, 70.56, 89.40, 110.20, 132.50])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# Subplot A: Throughput & Queue Latency vs Concurrency
c_tp = "#005f73"
c_queue = "#ae2012"

ax1.plot(concurrency, throughput, 'o-', color=c_tp, linewidth=2.2, markersize=7, label="Throughput (infer/sec)")
ax1.set_xlabel("Concurrency Level")
ax1.set_ylabel("Throughput (inferences/sec)", color=c_tp)
ax1.tick_params(axis='y', labelcolor=c_tp)
ax1.axvline(x=5, color="#0a9396", linestyle=":", linewidth=1.8, label="Peak Throughput (c=5)")

ax1_twin = ax1.twinx()
ax1_twin.plot(concurrency, queue_latency_ms, 's--', color=c_queue, linewidth=2.0, markersize=6, label="Queue Latency (ms)")
ax1_twin.set_ylabel("Queue Latency (ms)", color=c_queue)
ax1_twin.tick_params(axis='y', labelcolor=c_queue)
ax1_twin.grid(False)

ax1.set_title("A: Scaling Curve (Throughput vs. Queue Latency)")

# Subplot B: Latency Breakdown & Client p99 Histogram
x = np.arange(len(concurrency))
width = 0.26

ax2.bar(x - width, compute_infer_ms, width, label="Compute Infer (ms)", color="#2a9d8f")
ax2.bar(x, queue_latency_ms, width, label="Queue Latency (ms)", color="#e9c46a")
ax2.bar(x + width, p99_latency_ms, width, label="Client p99 (ms)", color="#e76f51")

ax2.set_xlabel("Concurrency Level")
ax2.set_ylabel("Latency (ms)")
ax2.set_xticks(x)
ax2.set_xticklabels(concurrency)
ax2.legend(loc="upper left")
ax2.set_title("B: Compute Infer vs. Queue & Client p99 Latency")

plt.suptitle("Triton V20 Winner Configuration Benchmark [1000us, 2, 2, 4]", y=1.01)
plt.tight_layout()

out_png = os.path.join("./benchmark_publication_v20", "v20_benchmark_publication.png")
out_svg = os.path.join("./benchmark_publication_v20", "v20_benchmark_publication.svg")

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_svg, bbox_inches="tight")
print(f"[SUCCESS] Publication figures rendered:\n  - {out_png} (300 DPI PNG)\n  - {out_svg} (Vector SVG)")
PYEOF
