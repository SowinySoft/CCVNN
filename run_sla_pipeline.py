import argparse
import asyncio
import csv
import json
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException


# ==============================================================================
# UTILS & WSL2 SAFE PATH SAVER
# ==============================================================================
def safe_save_figure(fig, target_path_str: str, dpi: int = 300) -> None:
    """
    Saves a Matplotlib figure safely across WSL2 mounted filesystems (/mnt/c/...)
    to prevent OSError: [Errno 22] Invalid argument issues caused by binary open modes.
    """
    target_path = Path(target_path_str).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if str(target_path).startswith("/mnt/"):
        tmp_path = Path("/tmp") / target_path.name
        fig.savefig(str(tmp_path), dpi=dpi)
        shutil.copyfile(str(tmp_path), str(target_path))
        if tmp_path.exists():
            tmp_path.unlink()
    else:
        fig.savefig(str(target_path), dpi=dpi)


# ==============================================================================
# 1. OPTIMIZED BENCHMARK ENGINE
# ==============================================================================
async def benchmark_concurrency(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    total_requests: int,
    concurrency: int,
) -> dict:
    """Runs an asynchronous benchmark using asyncio.Queue and lock-free worker metrics collection."""
    metadata = await client.get_model_metadata(model_name)

    # Pre-allocate reference NumPy arrays ONCE globally
    prebuilt_data = []
    for inp in metadata.inputs:
        shape = [dim if dim > 0 else 1 for dim in inp.shape]
        dtype = inp.datatype
        if "INT" in dtype:
            data = np.ones(shape, dtype=np.int32)
        elif "FP16" in dtype:
            data = np.ones(shape, dtype=np.float16)
        else:
            data = np.ones(shape, dtype=np.float32)
        prebuilt_data.append((inp.name, shape, dtype, data))

    # Populate task queue
    task_queue = asyncio.Queue()
    for i in range(total_requests):
        task_queue.put_nowait(i)

    # Track progress across workers safely
    completed_counter = 0
    progress_lock = asyncio.Lock()

    async def worker() -> list:
        nonlocal completed_counter
        local_latencies = []

        # PRE-ALLOCATE worker-local inputs once per worker task.
        worker_inputs = []
        for name, shape, dtype, data in prebuilt_data:
            infer_input = grpcclient.InferInput(name, shape, dtype)
            infer_input.set_data_from_numpy(data)
            worker_inputs.append(infer_input)

        while not task_queue.empty():
            try:
                task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            start_time = time.perf_counter()
            try:
                await client.infer(model_name=model_name, inputs=worker_inputs)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                local_latencies.append(latency_ms)

                async with progress_lock:
                    completed_counter += 1
                    if (
                        completed_counter % max(1, total_requests // 5) == 0
                        or completed_counter == total_requests
                    ):
                        print(
                            f"    Progress: {completed_counter}/{total_requests} finished...",
                            flush=True,
                        )
            except InferenceServerException as e:
                print(f"Inference error: {e}", flush=True)
            finally:
                task_queue.task_done()

        return local_latencies

    # Warmup using worker-local structures
    print(f"\n -> [Concurrency {concurrency:2d}] Running 5 warmup requests...", flush=True)
    warmup_inputs = []
    for name, shape, dtype, data in prebuilt_data:
        inp = grpcclient.InferInput(name, shape, dtype)
        inp.set_data_from_numpy(data)
        warmup_inputs.append(inp)

    for _ in range(5):
        await client.infer(model_name=model_name, inputs=warmup_inputs)

    print(f" -> [Concurrency {concurrency:2d}] Benchmarking {total_requests} requests...", flush=True)
    start_bench = time.perf_counter()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    results = await asyncio.gather(*workers)

    total_time = time.perf_counter() - start_bench

    # Flatten worker-local latencies
    latencies = [lat for worker_res in results for lat in worker_res]
    latencies_arr = np.array(latencies)
    throughput = len(latencies) / total_time if total_time > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": len(latencies),
        "total_time_sec": total_time,
        "throughput_fps": throughput,
        "mean_latency_ms": float(np.mean(latencies_arr)) if len(latencies_arr) > 0 else 0.0,
        "p50_latency_ms": float(np.percentile(latencies_arr, 50)) if len(latencies_arr) > 0 else 0.0,
        "p90_latency_ms": float(np.percentile(latencies_arr, 90)) if len(latencies_arr) > 0 else 0.0,
        "p95_latency_ms": float(np.percentile(latencies_arr, 95)) if len(latencies_arr) > 0 else 0.0,
        "p99_latency_ms": float(np.percentile(latencies_arr, 99)) if len(latencies_arr) > 0 else 0.0,
        "min_latency_ms": float(np.min(latencies_arr)) if len(latencies_arr) > 0 else 0.0,
        "max_latency_ms": float(np.max(latencies_arr)) if len(latencies_arr) > 0 else 0.0,
        "latencies_raw": latencies_arr.tolist(),
    }


# ==============================================================================
# 2. PLOTTING MODULE
# ==============================================================================
def generate_plots(data: dict, summary_img: str, dist_img: str, target_concurrency: int):
    """Generates throughput/latency plots and a latency distribution histogram."""
    runs = data["runs"]
    concurrencies = [r["concurrency"] for r in runs]
    throughputs = [r["throughput_fps"] for r in runs]
    p50 = [r["p50_latency_ms"] for r in runs]
    p90 = [r["p90_latency_ms"] for r in runs]
    p99 = [r["p99_latency_ms"] for r in runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"SLA Benchmark Performance: {data['model_name']}", fontsize=14, fontweight="bold")

    ax1.plot(concurrencies, throughputs, marker="o", color="#1f77b4", linewidth=2.5)
    ax1.set_title("Throughput vs. Concurrency", fontsize=12)
    ax1.set_xlabel("Concurrent Worker Tasks", fontsize=10)
    ax1.set_ylabel("Inferences / Second (FPS)", fontsize=10)
    ax1.set_xticks(concurrencies)
    ax1.grid(True, linestyle="--", alpha=0.6)
    for i, txt in enumerate(throughputs):
        ax1.annotate(f"{txt:.0f}", (concurrencies[i], throughputs[i]), textcoords="offset points", xytext=(0, 8), ha="center")

    ax2.plot(concurrencies, p50, marker="s", color="#2ca02c", linewidth=2, label="P50 (Median)")
    ax2.plot(concurrencies, p90, marker="^", color="#ff7f0e", linewidth=2, label="P90")
    ax2.plot(concurrencies, p99, marker="d", color="#d62728", linewidth=2, label="P99")
    ax2.set_title("Latency Percentiles vs. Concurrency", fontsize=12)
    ax2.set_xlabel("Concurrent Worker Tasks", fontsize=10)
    ax2.set_ylabel("Latency (ms)", fontsize=10)
    ax2.set_xticks(concurrencies)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    safe_save_figure(fig, summary_img, dpi=300)
    plt.close(fig)

    target_run = next((r for r in runs if r["concurrency"] == target_concurrency), None)
    if target_run and "latencies_raw" in target_run:
        latencies = target_run["latencies_raw"]
        fig_dist = plt.figure(figsize=(8, 5))
        plt.hist(latencies, bins=40, color="#4c72b0", edgecolor="black", alpha=0.7)
        plt.axvline(target_run["p50_latency_ms"], color="#2ca02c", linestyle="--", linewidth=2, label=f"P50: {target_run['p50_latency_ms']:.2f}ms")
        plt.axvline(target_run["p90_latency_ms"], color="#ff7f0e", linestyle="--", linewidth=2, label=f"P90: {target_run['p90_latency_ms']:.2f}ms")
        plt.axvline(target_run["p99_latency_ms"], color="#d62728", linestyle="--", linewidth=2, label=f"P99: {target_run['p99_latency_ms']:.2f}ms")
        plt.title(f"Latency Distribution (Concurrency = {target_concurrency})", fontsize=12, fontweight="bold")
        plt.xlabel("Latency (ms)", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        safe_save_figure(fig_dist, dist_img, dpi=300)
        plt.close(fig_dist)


# ==============================================================================
# 3. REPORT EXPORTER
# ==============================================================================
def export_reports(data: dict, csv_path: str, md_path: str):
    """Exports CSV metrics and Markdown SLA report."""
    runs = data["runs"]
    model_name = data["model_name"]

    headers = [
        "Concurrency", "Total_Requests", "Total_Time_Sec", "Throughput_FPS",
        "Mean_Latency_MS", "P50_Latency_MS", "P90_Latency_MS", "P95_Latency_MS",
        "P99_Latency_MS", "Min_Latency_MS", "Max_Latency_MS"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in runs:
            writer.writerow([
                r["concurrency"], r["total_requests"], f"{r['total_time_sec']:.4f}",
                f"{r['throughput_fps']:.2f}", f"{r['mean_latency_ms']:.3f}",
                f"{r['p50_latency_ms']:.3f}", f"{r['p90_latency_ms']:.3f}",
                f"{r['p95_latency_ms']:.3f}", f"{r['p99_latency_ms']:.3f}",
                f"{r['min_latency_ms']:.3f}", f"{r['max_latency_ms']:.3f}"
            ])

    p90_passed = all(r["p90_latency_ms"] <= 10.0 for r in runs)
    p99_passed = all(r["p99_latency_ms"] <= 20.0 for r in runs)

    md_content = f"# SLA Validation Report: `{model_name}`\n\n"
    md_content += "## 1. Executive Summary\n\n"
    md_content += f"- **Target Model:** `{model_name}`\n"
    md_content += f"- **Protocol:** Async gRPC\n"
    md_content += f"- **Tested Concurrencies:** {', '.join(str(r['concurrency']) for r in runs)}\n\n"
    md_content += "## 2. Benchmark Metrics\n\n"
    md_content += "| Concurrency | Throughput (infer/s) | Mean (ms) | P50 (ms) | P90 (ms) | P99 (ms) |\n"
    md_content += "|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    for r in runs:
        md_content += f"| {r['concurrency']} | {r['throughput_fps']:.2f} | {r['mean_latency_ms']:.2f} | {r['p50_latency_ms']:.2f} | {r['p90_latency_ms']:.2f} | {r['p99_latency_ms']:.2f} |\n"
    md_content += "\n## 3. SLA Compliance\n\n"
    md_content += f"- [{'x' if p90_passed else ' '}] **P90 Target (<= 10.0 ms)**\n"
    md_content += f"- [{'x' if p99_passed else ' '}] **P99 Target (<= 20.0 ms)**\n"

    with open(md_path, "w") as f:
        f.write(md_content)


# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================
async def main():
    parser = argparse.ArgumentParser(description="End-to-End Triton SLA Benchmark & Reporting Pipeline")
    parser.add_argument("--url", type=str, default="127.0.0.1:8001", help="Triton gRPC endpoint (Default: 127.0.0.1:8001)")
    parser.add_argument("--model", type=str, default="cyclotron_ensemble", help="Target model name")
    parser.add_argument("--requests", type=int, default=200, help="Total requests per run (Default: 200)")
    parser.add_argument("--concurrency-list", type=str, default="1,2,4,8,16,32", help="Concurrency levels")
    parser.add_argument("--out-json", type=str, default="benchmark_results.json", help="Output JSON path")
    parser.add_argument("--out-csv", type=str, default="benchmark_metrics.csv", help="Output CSV path")
    parser.add_argument("--out-md", type=str, default="SLA_Validation_Report.md", help="Output Markdown path")
    parser.add_argument("--out-plot", type=str, default="benchmark_summary.png", help="Summary plot path")
    parser.add_argument("--out-dist", type=str, default="latency_distribution.png", help="Distribution plot path")
    args = parser.parse_args()

    client = grpcclient.InferenceServerClient(url=args.url)
    try:
        if not await client.is_model_ready(args.model):
            raise RuntimeError(f"Model '{args.model}' is not ready on Triton gRPC server at {args.url}")

        concurrencies = [int(c.strip()) for c in args.concurrency_list.split(",")]
        pipeline_data = {"model_name": args.model, "timestamp": time.time(), "runs": []}

        print(f"\n[1/3] Running Async Benchmark Pipeline for model '{args.model}'...")
        print("=" * 65)

        for c in concurrencies:
            res = await benchmark_concurrency(client, args.model, args.requests, c)
            pipeline_data["runs"].append(res)
            print(f" -> Concurrency {c:2d} COMPLETE | Throughput: {res['throughput_fps']:7.2f} FPS | P50: {res['p50_latency_ms']:5.2f}ms | P99: {res['p99_latency_ms']:5.2f}ms\n")

        print("\n[2/3] Exporting Metrics & Reports...")
        with open(args.out_json, "w") as f:
            json.dump(pipeline_data, f, indent=2)
        export_reports(pipeline_data, args.out_csv, args.out_md)
        print(f" - Saved: {args.out_json}\n - Saved: {args.out_csv}\n - Saved: {args.out_md}")

        print("\n[3/3] Generating Visualizations...")
        target_c = 8 if 8 in concurrencies else concurrencies[0]
        generate_plots(pipeline_data, args.out_plot, args.out_dist, target_concurrency=target_c)
        print(f" - Saved: {args.out_plot}\n - Saved: {args.out_dist}")

        print("\nPipeline execution completed successfully!\n")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())