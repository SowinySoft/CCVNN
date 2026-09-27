import csv
import os

csv_path = "perf_sla_results.csv"
p90_limit = 10.0
p99_limit = 20.0

if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
    print("No CSV results recorded.")
    exit(0)

print(f"{'Concurrency':<12} | {'Throughput (FPS)':<18} | {'p50 (ms)':<10} | {'p90 (ms)':<10} | {'p99 (ms)':<10} | {'SLA Status'}")
print("-" * 80)

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            c = int(row["Concurrency"])
            fps = float(row["Inferences/Second"])
            p50 = float(row["p50 latency"]) / 1000.0
            p90 = float(row["p90 latency"]) / 1000.0
            p99 = float(row["p99 latency"]) / 1000.0
            status = "PASS" if p90 <= p90_limit and p99 <= p99_limit else "FAIL (Breached)"
            print(f"{c:<12} | {fps:<18.2f} | {p50:<10.2f} | {p90:<10.2f} | {p99:<10.2f} | {status}")
        except Exception:
            pass
