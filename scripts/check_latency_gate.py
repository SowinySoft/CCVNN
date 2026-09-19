import json
import os
import sys

json_path = os.getenv("BENCHMARK_RESULTS_PATH", "benchmark_results.json")

if not os.path.exists(json_path):
    print(f"[ERROR] Benchmark results file '{json_path}' not found.")
    sys.exit(1)

try:
    with open(json_path, "r") as f:
        data = json.load(f)
except Exception as e:
    print(f"[ERROR] Failed to parse JSON from '{json_path}': {e}")
    sys.exit(1)

# Safely traverse metrics structure
summary = data.get("summary_ms", {}).get("total_glass_to_glass", {})
p99 = summary.get("p99")
p50 = summary.get("p50")
mean = summary.get("mean")

if p99 is None or p50 is None:
    print(f"[ERROR] Invalid structure in '{json_path}'. Missing 'p99' or 'p50' metrics.")
    sys.exit(1)

# Environment detection and SLA SLA threshold calculation
IS_CLOUD_CI = os.getenv("CI", "false").lower() in ("true", "1") or os.path.exists("/content")
default_threshold = 10.0 if IS_CLOUD_CI else 2.0
MAX_P99_THRESHOLD = float(os.getenv("LATENCY_P99_THRESHOLD", default_threshold))

mean_str = f" | mean: {mean:.3f} ms" if mean is not None else ""
print(f"[INFO] Metric Check -> p50: {p50:.3f} ms | p99: {p99:.3f} ms{mean_str} (Target: < {MAX_P99_THRESHOLD:.1f} ms)")

if p99 >= MAX_P99_THRESHOLD:
    print(f"[ERROR] SLA Latency Gate Violation! p99 ({p99:.3f} ms) exceeds limit of {MAX_P99_THRESHOLD:.1f} ms.")
    sys.exit(1)

run_environment = "Cloud Synthetic Run (CI)" if IS_CLOUD_CI else "Production Edge Target"
print(f"[✓] SLA Gate Passed successfully for {run_environment}!")