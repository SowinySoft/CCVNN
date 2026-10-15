import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual theme
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Triton OpenVINO Benchmark Comparison: Full Model (ccvnn) vs. Backbone (ccvnn_v20_backbone)\n"
    "Configuration: [max_queue_delay: 2000µs | instances: 2 | streams: 2 | threads: 4]",
    fontsize=13,
    fontweight="bold",
)

# Load CSV files or fallback to parsed benchmark metrics
files = {
    "ccvnn (Full)": "perf_ccvnn_dynamic_batching.csv",
    "ccvnn_v20_backbone": "perf_ccvnn_v20_backbone_dynamic_batching.csv",
}

colors = {"ccvnn (Full)": "#D9381E", "ccvnn_v20_backbone": "#0072B2"}

# Raw benchmark fallback dictionaries if CSV header mapping varies
data_backup = {
    "ccvnn (Full)": {
        "concurrency": [1, 3, 5, 7],
        "throughput": [262.384, 277.989, 436.010, 435.939],
        "avg_lat": [3.221, 8.553, 9.796, 14.949],
        "p90_lat": [4.119, 9.735, 14.280, 24.527],
        "p95_lat": [5.403, 15.149, 20.056, 32.439],
        "p99_lat": [12.574, 108.344, 71.072, 67.869],
        "queue_time": [0.082, 0.833, 2.294, 5.639],
        "infer_time": [1.598, 3.967, 2.884, 3.231],
    },
    "ccvnn_v20_backbone": {
        "concurrency": [1, 3, 5, 7],
        "throughput": [321.192, 475.640, 442.810, 570.007],
        "avg_lat": [2.906, 5.777, 10.195, 11.380],
        "p90_lat": [3.896, 8.749, 15.370, 15.047],
        "p95_lat": [4.350, 11.029, 19.689, 19.003],
        "p99_lat": [7.113, 20.558, 46.511, 51.825],
        "queue_time": [0.082, 0.542, 2.387, 4.555],
        "infer_time": [1.339, 2.446, 2.940, 2.447],
    },
}

# ---------------------------------------------------------
# Plot 1: Throughput vs. Concurrency (Curved Dots)
# ---------------------------------------------------------
ax1 = axes[0, 0]
for model, d in data_backup.items():
    ax1.plot(
        d["concurrency"],
        d["throughput"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=model,
        color=colors[model],
    )
    for x, y in zip(d["concurrency"], d["throughput"]):
        ax1.annotate(
            f"{y:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

ax1.set_title("Throughput Scaling (Inferences / sec)", fontweight="bold")
ax1.set_xlabel("Concurrency Level")
ax1.set_ylabel("Inferences / Second")
ax1.set_xticks([1, 3, 5, 7])
ax1.legend(loc="upper left")

# ---------------------------------------------------------
# Plot 2: Average & p95 Latency vs. SLA Target
# ---------------------------------------------------------
ax2 = axes[0, 1]
for model, d in data_backup.items():
    ax2.plot(
        d["concurrency"],
        d["avg_lat"],
        marker="o",
        linewidth=2,
        label=f"{model} (Avg)",
        color=colors[model],
    )
    ax2.plot(
        d["concurrency"],
        d["p95_lat"],
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label=f"{model} (p95)",
        color=colors[model],
        alpha=0.7,
    )

ax2.axhline(
    y=10.0,
    color="black",
    linestyle=":",
    linewidth=2,
    label="10ms SLA Target Limit",
)
ax2.set_title(
    "Client Latency (Avg & p95) vs. 10ms SLA Target", fontweight="bold"
)
ax2.set_xlabel("Concurrency Level")
ax2.set_ylabel("Latency (ms)")
ax2.set_xticks([1, 3, 5, 7])
ax2.legend(loc="upper left", fontsize=8)

# ---------------------------------------------------------
# Plot 3: Latency Percentile Grouped Bar Chart at C=3
# ---------------------------------------------------------
ax3 = axes[1, 0]
labels = ["p50", "p90", "p95", "p99"]
x = np.arange(len(labels))
width = 0.35

c3_ccvnn = [4.625, 9.735, 15.149, 108.344]
c3_backbone = [4.526, 8.749, 11.029, 20.558]

rects1 = ax3.bar(
    x - width / 2,
    c3_ccvnn,
    width,
    label="ccvnn (Full)",
    color=colors["ccvnn (Full)"],
)
rects2 = ax3.bar(
    x + width / 2,
    c3_backbone,
    width,
    label="ccvnn_v20_backbone",
    color=colors["ccvnn_v20_backbone"],
)

ax3.axhline(y=10.0, color="black", linestyle=":", linewidth=1.5)
ax3.set_title(
    "Latency Distribution at Operational Concurrency (C=3)", fontweight="bold"
)
ax3.set_ylabel("Latency (ms)")
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.set_yscale("log")  # Log scale to cleanly accommodate p99 tail spike
ax3.legend()

# ---------------------------------------------------------
# Plot 4: Server Time Decomposition (Queue vs Compute Infer)
# ---------------------------------------------------------
ax4 = axes[1, 1]
concurrency_labels = ["C=1", "C=3", "C=5", "C=7"]
x_arr = np.arange(len(concurrency_labels))

ax4.bar(
    x_arr - width / 2,
    data_backup["ccvnn (Full)"]["queue_time"],
    width,
    label="ccvnn Queue",
    color="#D9381E",
    alpha=0.5,
)
ax4.bar(
    x_arr - width / 2,
    data_backup["ccvnn (Full)"]["infer_time"],
    width,
    bottom=data_backup["ccvnn (Full)"]["queue_time"],
    label="ccvnn Compute",
    color="#D9381E",
)

ax4.bar(
    x_arr + width / 2,
    data_backup["ccvnn_v20_backbone"]["queue_time"],
    width,
    label="Backbone Queue",
    color="#0072B2",
    alpha=0.5,
)
ax4.bar(
    x_arr + width / 2,
    data_backup["ccvnn_v20_backbone"]["infer_time"],
    width,
    bottom=data_backup["ccvnn_v20_backbone"]["queue_time"],
    label="Backbone Compute",
    color="#0072B2",
)

ax4.set_title("Server Time Breakdown (Queue vs. Compute)", fontweight="bold")
ax4.set_ylabel("Time (ms)")
ax4.set_xticks(x_arr)
ax4.set_xticklabels(concurrency_labels)
ax4.legend(fontsize=8)

plt.tight_layout()
output_path = "triton_benchmark_results.png"
plt.savefig(output_path, dpi=300)
print(f"[SUCCESS] Benchmark visualization saved to {output_path}")