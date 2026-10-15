#!/usr/bin/env python3
import csv
import glob
import os
import re


def safe_float(val, default=0.0):
    """Safely convert value or string to float, stripping non-numeric characters if needed."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    # Extract first numeric float match from string (handles '123.45 us', '', 'N/A')
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))
    return float(match.group(0)) if match else default


def find_column_val(row, candidate_patterns):
    """
    Search DictReader row for column names matching candidate regex patterns
    (case-insensitive, ignoring trailing unit bracket annotations like '(us)').
    """
    # Create normalized map of row keys
    normalized_row = {
        re.sub(r"\s+", " ", k.strip().lower()): v
        for k, v in row.items()
        if k
    }

    for pattern in candidate_patterns:
        pattern_clean = pattern.lower()
        # Direct match check
        if pattern_clean in normalized_row:
            return normalized_row[pattern_clean]
        # Partial regex match
        for norm_key, val in normalized_row.items():
            if re.search(pattern_clean, norm_key):
                return val
    return None


def parse_perf_csv(file_path):
    """Parse a single perf_analyzer CSV report and extract key metrics."""
    filename = os.path.basename(file_path)

    # Extract model name and batch size from filename: perf_{model}_batch_{batch}.csv
    match = re.match(r"perf_(.+)_batch_(\d+)\.csv", filename)
    if match:
        model_name, batch_size_str = match.group(1), match.group(2)
        batch_size = int(batch_size_str)
    else:
        model_name = filename.replace(".csv", "").replace("perf_", "")
        batch_size = 1

    records = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Fuzzy match header columns across different perf_analyzer versions
            raw_conc = find_column_val(
                row, ["concurrency", "conc", "client concurrency"]
            )
            raw_tp = find_column_val(
                row, ["inferences/second", "throughput", "inferences_per_second"]
            )
            raw_avg = find_column_val(
                row, ["avg latency", "client latency", "avg_latency", "p50 latency"]
            )
            raw_p95 = find_column_val(
                row, ["p95 latency", "p95_latency", "95th percentile"]
            )
            raw_queue = find_column_val(
                row, ["server queue", "queue latency", "server_queue"]
            )
            raw_compute = find_column_val(
                row, ["server compute infer", "compute latency", "server_compute"]
            )

            concurrency = int(safe_float(raw_conc, default=1))
            throughput = safe_float(raw_tp)

            # Latency values in perf_analyzer CSV are reported in microseconds (us) -> convert to ms
            avg_latency_ms = safe_float(raw_avg) / 1000.0
            p95_latency_ms = safe_float(raw_p95) / 1000.0
            queue_ms = safe_float(raw_queue) / 1000.0
            compute_ms = safe_float(raw_compute) / 1000.0

            # If client avg latency is missing, fallback to queue + compute total
            if avg_latency_ms == 0.0 and (queue_ms > 0 or compute_ms > 0):
                avg_latency_ms = queue_ms + compute_ms

            # Compute actual samples processed per second
            sample_throughput = throughput * batch_size

            records.append(
                {
                    "model": model_name,
                    "batch_size": batch_size,
                    "concurrency": concurrency,
                    "throughput": throughput,
                    "sample_throughput": sample_throughput,
                    "avg_latency_ms": avg_latency_ms,
                    "p95_latency_ms": p95_latency_ms,
                    "compute_ms": compute_ms,
                    "queue_ms": queue_ms,
                }
            )
    return records


def print_comparison_table(records):
    """Render a clean ASCII comparison table sorted by Model, Batch, and Concurrency."""
    if not records:
        print("[-] No performance CSV data found.")
        return

    # Sort logically: Model -> Batch Size -> Concurrency
    records.sort(
        key=lambda r: (
            r["model"],
            r["batch_size"] if isinstance(r["batch_size"], int) else 0,
            r["concurrency"] if isinstance(r["concurrency"], int) else 0,
        )
    )

    header = (
        f"| {'Model Name':<20} | {'Batch':<5} | {'Conc':<4} | "
        f"{'Throughput (inf/s)':<18} | {'Samples/sec':<12} | "
        f"{'Avg Lat (ms)':<12} | {'p95 Lat (ms)':<12} | {'Compute (ms)':<12} | {'Queue (ms)':<10} |"
    )
    divider = "-" * len(header)

    print("\n" + divider)
    print(" TRITON INFERENCE PERFORMANCE SUMMARY TABLE")
    print(divider)
    print(header)
    print(divider)

    current_model = None
    for r in records:
        if current_model and current_model != r["model"]:
            print(divider)
        current_model = r["model"]

        print(
            f"| {r['model']:<20} | {r['batch_size']:<5} | {r['concurrency']:<4} | "
            f"{r['throughput']:<18.2f} | {r['sample_throughput']:<12.2f} | "
            f"{r['avg_latency_ms']:<12.2f} | {r['p95_latency_ms']:<12.2f} | "
            f"{r['compute_ms']:<12.2f} | {r['queue_ms']:<10.2f} |"
        )

    print(divider + "\n")


def export_consolidated_csv(records, output_path="triton_summary_results.csv"):
    """Export parsed records to a single clean CSV file for plotting utilities."""
    if not records:
        return
    fieldnames = [
        "model",
        "batch_size",
        "concurrency",
        "throughput",
        "sample_throughput",
        "avg_latency_ms",
        "p95_latency_ms",
        "compute_ms",
        "queue_ms",
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[+] Consolidated summary saved to '{output_path}'")


def main():
    csv_files = glob.glob("perf_*.csv")
    if not csv_files:
        print("[-] No 'perf_*.csv' files found in current directory.")
        return

    all_records = []
    for csv_file in csv_files:
        try:
            records = parse_perf_csv(csv_file)
            all_records.extend(records)
        except Exception as e:
            print(f"[!] Warning: Failed to parse {csv_file}: {e}")

    print_comparison_table(all_records)
    export_consolidated_csv(all_records)


if __name__ == "__main__":
    main()