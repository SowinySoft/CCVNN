import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
df = pd.read_csv("benchmark_4way_results.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Triton OpenVINO Benchmark Comparison: CCVNN Variants vs YOLOv8 Baselines", fontsize=13, fontweight="bold")

# Palette updated to include all 6 models present in the CSV
palette = {
    "CCVNN (Full)": "#D9381E",
    "CCVNN (Backbone)": "#0072B2",
    "CCVNN v21 Dynamic": "#009E73",
    "CCVNN v21 INT8": "#CC79A7",
    "YOLOv8n Baseline": "#E69F00",
    "YOLOv8n OpenVINO": "#D55E00"
}

# Plot 1: Throughput Comparison
sns.lineplot(
    data=df, x="Concurrency", y="Throughput", hue="Model",
    marker="o", linewidth=2.5, markersize=8, palette=palette, ax=axes[0]
)
axes[0].set_title("Throughput Scaling (Inferences / sec)", fontweight="bold")
axes[0].set_ylabel("Throughput (infer/sec)")
axes[0].set_xticks([1, 3, 5, 7])

# Plot 2: p95 Latency vs SLA Target
sns.lineplot(
    data=df, x="Concurrency", y="p95_Latency_ms", hue="Model",
    marker="s", linewidth=2.5, markersize=8, palette=palette, ax=axes[1]
)
axes[1].axhline(y=10.0, color="black", linestyle=":", linewidth=2, label="10ms SLA Limit")
axes[1].set_title("p95 Latency vs. 10ms SLA Target", fontweight="bold")
axes[1].set_ylabel("p95 Latency (ms)")
axes[1].set_xticks([1, 3, 5, 7])

plt.tight_layout()
plt.savefig("benchmark_4way_results.png", dpi=300)
print("[SUCCESS] Dashboard saved to benchmark_4way_results.png")