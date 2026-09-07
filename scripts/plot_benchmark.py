import json
import os
import matplotlib.pyplot as plt

def generate_benchmark_plot():
    json_path = "benchmark_results.json"
    if not os.path.exists(json_path):
        print(f"[!] {json_path} not found, skipping plot generation.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    summary = data.get("summary_ms", {})
    if not summary:
        print("[!] No summary data found in benchmark_results.json")
        return

    # Dynamically extract present stages
    keys = [k for k in summary.keys() if isinstance(summary[k], dict) and "p50" in summary[k]]
    
    stages = [k.replace("_", " ").title() for k in keys]
    p50 = [summary[k]["p50"] for k in keys]
    p99 = [summary[k]["p99"] for k in keys]

    x = range(len(stages))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - width/2 for i in x], p50, width, label='p50 Latency (ms)', color='#2b5c8f')
    plt.bar([i + width/2 for i in x], p99, width, label='p99 Latency (ms)', color='#d9534f')

    plt.xlabel('Pipeline Stage')
    plt.ylabel('Latency (ms)')
    plt.title('CCVNN Stage Latency Profile')
    plt.xticks(x, stages, rotation=15)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    os.makedirs("public", exist_ok=True)
    plt.savefig("public/benchmark_plot.png", dpi=300)
    print("[✓] Benchmark plot generated at public/benchmark_plot.png")

if __name__ == "__main__":
    generate_benchmark_plot()
