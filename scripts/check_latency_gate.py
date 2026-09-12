import json, os, sys

json_path = 'benchmark_results.json'

if not os.path.exists(json_path):
    print(f"[ERROR] '{json_path}' not found.")
    sys.exit(1)

with open(json_path) as f:
    data = json.load(f)

p99 = data['summary_ms']['total_glass_to_glass']['p99']
p50 = data['summary_ms']['total_glass_to_glass']['p50']

IS_CLOUD_CI = os.getenv("CI", "false") == "true" or os.path.exists("/content")
MAX_P99_THRESHOLD = 10.0 if IS_CLOUD_CI else 2.0

print(f"[INFO] Metric Check -> p50: {p50:.3f} ms | p99: {p99:.3f} ms (Target: < {MAX_P99_THRESHOLD:.1f} ms)")

if p99 >= MAX_P99_THRESHOLD:
    print(f"[ERROR] Latency Gate Violation! p99 ({p99:.3f} ms) exceeds {MAX_P99_THRESHOLD} ms limit.")
    sys.exit(1)

print(f"[✓] Gate Passed for {'Cloud Synthetic Run' if IS_CLOUD_CI else 'Production Edge Target'}!")
