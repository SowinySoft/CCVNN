import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure publication-grade aesthetics
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.autolayout': True
})

sns.set_theme(style="whitegrid")

# 1. Load benchmark results
df = pd.read_csv("benchmark_4way_results.csv")

# Color mapping updated for all 6 models
model_colors = {
    "CCVNN (Backbone)": "#2ca02c",    # Green
    "CCVNN v21 Dynamic": "#ff7f0e",  # Orange
    "CCVNN v21 INT8": "#9467bd",     # Purple
    "CCVNN (Full)": "#1f77b4",       # Blue
    "YOLOv8n Baseline": "#d62728",   # Red
    "YOLOv8n OpenVINO": "#8c564b"    # Brown
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# 2. Subplot A: Throughput Scaling vs Concurrency
sns.lineplot(
    data=df,
    x="Concurrency",
    y="Throughput",
    hue="Model",
    style="Model",
    markers=True,
    dashes=False,
    markersize=9,
    linewidth=2.5,
    palette=model_colors,
    ax=axes[0]
)

axes[0].set_title("A) Throughput Scaling across Concurrency", fontweight="bold", pad=12)
axes[0].set_xlabel("Request Concurrency ($C$)")
axes[0].set_ylabel("Throughput (inferences / sec)")
axes[0].set_xticks([1, 3, 5, 7])
axes[0].grid(True, linestyle="--", alpha=0.5)

# 3. Subplot B: Average Latency vs Concurrency (Log Scale for Disparity)
sns.lineplot(
    data=df,
    x="Concurrency",
    y="Avg_Latency_ms",
    hue="Model",
    style="Model",
    markers=True,
    dashes=False,
    markersize=9,
    linewidth=2.5,
    palette=model_colors,
    ax=axes[1]
)

axes[1].set_yscale("log")
axes[1].set_title("B) Average Latency (Log Scale)", fontweight="bold", pad=12)
axes[1].set_xlabel("Request Concurrency ($C$)")
axes[1].set_ylabel("Latency (ms) [Log Scale]")
axes[1].set_xticks([1, 3, 5, 7])
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

# Add p95 latency error ranges
for model in df["Model"].unique():
    sub = df[df["Model"] == model]
    axes[1].vlines(
        x=sub["Concurrency"],
        ymin=sub["Avg_Latency_ms"],
        ymax=sub["p95_Latency_ms"],
        colors=model_colors[model],
        alpha=0.4,
        linewidth=1.8,
        label="_nolegend_"
    )

# Clean up redundant sub-legends and display a unified top legend
axes[0].get_legend().remove()
axes[1].get_legend().remove()

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, 
    labels, 
    loc="upper center", 
    bbox_to_anchor=(0.5, 1.12), 
    ncol=3, 
    frameon=True, 
    facecolor="white", 
    edgecolor="none"
)

# 4. Save Figures (PNG + SVG)
plt.savefig("triton_benchmark_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig("triton_benchmark_comparison.svg", format="svg", bbox_inches="tight")

print("Publication figures exported successfully:")
print("  - triton_benchmark_comparison.png (300 DPI raster)")
print("  - triton_benchmark_comparison.svg (Vector graphics)")