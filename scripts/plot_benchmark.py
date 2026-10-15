import json
import os
import matplotlib.pyplot as plt

def generate_benchmark_plot():
    json_path = os.getenv("BENCHMARK_RESULTS_PATH", "benchmark_results.json")
    if not os.path.exists(json_path):
        print(f"[!] {json_path} not found, skipping plot generation.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    summary = data.get("summary_ms", {})
    if not summary:
        print(f"[!] No summary data found in {json_path}")
        return

    # Dynamically extract present stages
    keys = [k for k in summary.keys() if isinstance(summary[k], dict) and "p50" in summary[k]]
    if not keys:
        print("[!] No valid stage latency metrics found.")
        return

    stages = [k.replace("_", " ").title() for k in keys]
    p50 = [summary[k]["p50"] for k in keys]
    p99 = [summary[k]["p99"] for k in keys]

    x = range(len(stages))
    width = 0.35

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))

    rects1 = ax.bar([i - width/2 for i in x], p50, width, label='p50 Latency (ms)', color='#00CC96')
    rects2 = ax.bar([i + width/2 for i in x], p99, width, label='p99 Latency (ms)', color='#EF553B')

    ax.set_xlabel('Pipeline Stage', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Latency (ms)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('CCVNN V14 Stage Latency Profile (14D Cumulative Vector Engine)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Value label annotations
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color='#E0E0E0')

    add_labels(rects1)
    add_labels(rects2)

    plt.tight_layout()

    os.makedirs("public", exist_ok=True)
    output_path = "public/benchmark_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] V14 Benchmark plot successfully generated at {output_path}")

if __name__ == "__main__":
    generate_benchmark_plot()