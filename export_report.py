# export_report.py
import argparse
import csv
import json
import os


def export_csv(runs: list, csv_path: str):
    """Exports raw benchmark run metrics into a structured CSV file."""
    headers = [
        "Concurrency",
        "Total_Requests",
        "Total_Time_Sec",
        "Throughput_FPS",
        "Mean_Latency_MS",
        "P50_Latency_MS",
        "P90_Latency_MS",
        "P95_Latency_MS",
        "P99_Latency_MS",
        "Min_Latency_MS",
        "Max_Latency_MS",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in runs:
            writer.writerow(
                [
                    r["concurrency"],
                    r["total_requests"],
                    f"{r['total_time_sec']:.4f}",
                    f"{r['throughput_fps']:.2f}",
                    f"{r['mean_latency_ms']:.3f}",
                    f"{r['p50_latency_ms']:.3f}",
                    f"{r['p90_latency_ms']:.3f}",
                    f"{r['p95_latency_ms']:.3f}",
                    f"{r['p99_latency_ms']:.3f}",
                    f"{r['min_latency_ms']:.3f}",
                    f"{r['max_latency_ms']:.3f}",
                ]
            )


def export_markdown(data: dict, md_path: str):
    """Generates an executive SLA Validation Report in Markdown format."""
    model_name = data.get("model_name", "N/A")
    runs = data.get("runs", [])

    md_content = f"# SLA Validation Report: `{model_name}`\n\n"
    md_content += "## 1. Executive Summary\n\n"
    md_content += f"- **Target Model:** `{model_name}`\n"
    md_content += f"- **Protocol:** Async gRPC\n"
    md_content += f"- **Evaluated Concurrency Levels:** {', '.join(str(r['concurrency']) for r in runs)}\n\n"

    md_content += "## 2. Performance Summary Table\n\n"
    md_content += (
        "| Concurrency | Throughput (infer/s) | Mean Latency (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Min/Max (ms) |\n"
    )
    md_content += "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"

    for r in runs:
        md_content += (
            f"| {r['concurrency']} "
            f"| {r['throughput_fps']:.2f} "
            f"| {r['mean_latency_ms']:.2f} "
            f"| {r['p50_latency_ms']:.2f} "
            f"| {r['p90_latency_ms']:.2f} "
            f"| {r['p99_latency_ms']:.2f} "
            f"| {r['min_latency_ms']:.2f} / {r['max_latency_ms']:.2f} |\n"
        )

    md_content += "\n## 3. SLA Compliance Checklist\n\n"
    
    # Example SLA thresholds (P90 < 10ms, P99 < 20ms)
    p90_passed = all(r["p90_latency_ms"] <= 10.0 for r in runs)
    p99_passed = all(r["p99_latency_ms"] <= 20.0 for r in runs)

    md_content += f"- [{'x' if p90_passed else ' '}] **P90 Latency SLA Target (<= 10.0 ms)**\n"
    md_content += f"- [{'x' if p99_passed else ' '}] **P99 Latency SLA Target (<= 20.0 ms)**\n"

    with open(md_path, "w") as f:
        f.write(md_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Triton Benchmark Metrics to CSV & Markdown")
    parser.add_argument("--json", type=str, default="benchmark_results.json", help="Input JSON file")
    parser.add_argument("--csv", type=str, default="benchmark_metrics.csv", help="Output CSV file path")
    parser.add_argument("--md", type=str, default="SLA_Validation_Report.md", help="Output Markdown report path")
    args = parser.parse_args()

    if not os.path.exists(args.json):
        raise FileNotFoundError(f"Source JSON file '{args.json}' not found.")

    with open(args.json, "r") as f:
        data = json.load(f)

    export_csv(data["runs"], args.csv)
    export_markdown(data, args.md)

    print(f"Exported CSV metrics to: '{args.csv}'")
    print(f"Exported Markdown report to: '{args.md}'")