#!/usr/bin/env python3
import argparse
import asyncio
import csv
import os
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException


async def send_async_request(client, semaphore, req_id, input_data, model_name, input_name, output_name):
    """Sends a single async request bounded by a concurrency semaphore."""
    async with semaphore:
        inputs = [grpcclient.InferInput(input_name, input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput(output_name)]

        start_time = time.perf_counter()
        try:
            response = await client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            _ = response.as_numpy(output_name)
            return req_id, latency_ms, "SUCCESS"
        except Exception as e:
            return req_id, None, f"FAILED: {str(e)}"


async def run_benchmark(args):
    client = grpcclient.InferenceServerClient(url=args.url)

    try:
        if not await client.is_server_ready():
            print(f"[ERROR] Triton server at '{args.url}' is not ready.")
            return
        if not await client.is_model_ready(args.model_name):
            print(f"[ERROR] Model '{args.model_name}' is not loaded/ready.")
            return
    except Exception as e:
        print(f"[ERROR] Could not connect to Triton server at {args.url}: {e}")
        return

    input_data = np.random.randn(args.batch_size, args.vector_dim).astype(np.float32)
    semaphore = asyncio.Semaphore(args.concurrency)

    print("========================================================")
    print(" CCVNN Async gRPC Benchmark CLI Tool")
    print(f" Target Server : {args.url}")
    print(f" Model Name    : {args.model_name}")
    print(f" Total Requests: {args.requests}")
    print(f" Concurrency   : {args.concurrency}")
    print(f" Payload Shape : ({args.batch_size}, {args.vector_dim}) [FP32]")
    if args.csv:
        print(f" CSV Export    : {args.csv}")
    print("========================================================\n")

    tasks = [
        send_async_request(
            client=client,
            semaphore=semaphore,
            req_id=i + 1,
            input_data=input_data,
            model_name=args.model_name,
            input_name=args.input_name,
            output_name=args.output_name
        )
        for i in range(args.requests)
    ]

    wall_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall_duration = time.perf_counter() - wall_start

    # Metrics computation
    latencies = [r[1] for r in results if r[2] == "SUCCESS"]
    failures = sum(1 for r in results if r[2] != "SUCCESS")

    if latencies:
        lat_arr = np.array(latencies)
        total_inferences = len(lat_arr) * args.batch_size
        throughput = total_inferences / wall_duration
        avg_lat = np.mean(lat_arr)
        min_lat = np.min(lat_arr)
        max_lat = np.max(lat_arr)
        p50 = np.percentile(lat_arr, 50)
        p95 = np.percentile(lat_arr, 95)
        p99 = np.percentile(lat_arr, 99)

        print("---------------- Benchmark Results ----------------")
        print(f" Total Requests Sent : {args.requests}")
        print(f" Successful Requests : {len(lat_arr)}")
        print(f" Failed Requests     : {failures}")
        print(f" Total Inferences    : {total_inferences:,}")
        print(f" Total Elapsed Time  : {wall_duration:.3f} s")
        print(f" System Throughput   : {throughput:,.2f} infer/sec")
        print(f" Avg Client Latency  : {avg_lat:.2f} ms")
        print(f" Min Latency         : {min_lat:.2f} ms")
        print(f" Max Latency         : {max_lat:.2f} ms")
        print(f" p50 Latency         : {p50:.2f} ms")
        print(f" p95 Latency         : {p95:.2f} ms")
        print(f" p99 Latency         : {p99:.2f} ms")
        print("---------------------------------------------------")

        # Export CSV
        if args.csv:
            csv_path = os.path.abspath(args.csv)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                
                # Summary header section
                writer.writerow(["# BENCHMARK CONFIG & SUMMARY METRICS"])
                writer.writerow(["model_name", args.model_name])
                writer.writerow(["concurrency", args.concurrency])
                writer.writerow(["batch_size", args.batch_size])
                writer.writerow(["total_requests", args.requests])
                writer.writerow(["successful_requests", len(lat_arr)])
                writer.writerow(["failed_requests", failures])
                writer.writerow(["throughput_infer_per_sec", round(throughput, 2)])
                writer.writerow(["avg_latency_ms", round(avg_lat, 2)])
                writer.writerow(["p50_latency_ms", round(p50, 2)])
                writer.writerow(["p95_latency_ms", round(p95, 2)])
                writer.writerow(["p99_latency_ms", round(p99, 2)])
                writer.writerow([])  # Blank spacer
                
                # Per-request dataset for plotting
                writer.writerow(["request_id", "latency_ms", "status", "concurrency", "batch_size"])
                for req_id, lat, status in results:
                    lat_str = f"{lat:.4f}" if lat is not None else ""
                    writer.writerow([req_id, lat_str, status, args.concurrency, args.batch_size])

            print(f"[SUCCESS] Benchmark metrics and raw dataset exported to: {csv_path}")

    await client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Async gRPC CLI benchmark tool for Triton Inference Server with CSV export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-c", "--concurrency", type=int, default=5,
                        help="Maximum number of concurrent async requests.")
    parser.add_argument("-b", "--batch-size", type=int, default=8,
                        help="Batch size per inference request.")
    parser.add_argument("-n", "--requests", type=int, default=100,
                        help="Total number of inference requests to send.")
    parser.add_argument("-o", "--csv", type=str, default=None,
                        help="Path to save output benchmark CSV file (e.g., results.csv).")

    # Server parameters
    parser.add_argument("-u", "--url", type=str, default="localhost:8001",
                        help="Triton gRPC server URL.")
    parser.add_argument("-m", "--model-name", type=str, default="ccvnn_v20_backbone",
                        help="Name of the model hosted on Triton.")
    parser.add_argument("-d", "--vector-dim", type=int, default=512,
                        help="Input vector feature dimension.")
    parser.add_argument("--input-name", type=str, default="input",
                        help="Input tensor name in config.pbtxt.")
    parser.add_argument("--output-name", type=str, default="output",
                        help="Output tensor name in config.pbtxt.")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()