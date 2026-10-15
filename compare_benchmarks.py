import csv

def read_perf(file_path):
    results = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[int(row['Concurrency'])] = (float(row['Throughput']), float(row['Avg Latency']))
    return results

fp32 = read_perf('benchmark_dynamic_fp32.csv')
int8 = read_perf('benchmark_int8.csv')

print("\n=== PERFORMANCE COMPARISON (FP32 vs INT8) ===")
print(f"{'Concurrency':<12} | {'FP32 Infer/s':<12} | {'INT8 Infer/s':<12} | {'Speedup':<10}")
print("-" * 55)

for c in sorted(fp32.keys()):
    fp32_tp, _ = fp32[c]
    int8_tp, _ = int8.get(c, (0, 0))
    speedup = (int8_tp / fp32_tp) if fp32_tp > 0 else 0
    print(f"{c:<12} | {fp32_tp:<12.2f} | {int8_tp:<12.2f} | {speedup:.2f}x")