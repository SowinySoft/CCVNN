# async_benchmark.py
import argparse
import asyncio
import json
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException


async def benchmark_concurrency(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    total_requests: int,
    concurrency: int,
) -> dict:
    """Runs an asynchronous benchmark for a specific concurrency level."""
    # Infer input details from Triton metadata
    metadata = await client.get_model_metadata(model_name)
    inputs = []
    for inp in metadata.inputs:
        shape = [dim if dim > 0 else 1 for dim in inp.shape]
        dtype = inp.datatype
        if "INT" in dtype:
            data = np.ones(shape, dtype=np.int32)
        elif "FP16" in dtype:
            data = np.ones(shape, dtype=np.float16)
        else:
            data = np.ones(shape, dtype=np.float32)

        infer_input = grpcclient.InferInput(inp.name, shape, dtype)
        infer_input.set_data_from_numpy(data)
        inputs.append(infer_input)

    queue = asyncio.Queue()
    for _ in range(total_requests):
        queue.put_nowait(True)

    latencies = []
    lock = asyncio.Lock()

    async def worker():
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            start_time = time.perf_counter()
            try:
                await client.infer(model_name=model_name, inputs=inputs)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                async with lock:
                    latencies.append(latency_ms)
            except InferenceServerException as e:
                print(f"Inference error: {e}")
            finally:
                queue.task_done()

    # Warmup
    warmup_inputs = inputs
    for _ in range(10):
        await client.infer(model_name=model_name, inputs=warmup_inputs)

    start_bench = time.perf_counter()
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)
    total_time = time.perf_counter() - start_bench

    latencies_arr = np.array(latencies)
    throughput = len(latencies) / total_time if total_time > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": len(latencies),
        "total_time_sec": total_time,
        "throughput_fps": throughput,
        "mean_latency_ms": float(np.mean(latencies_arr)),
        "p50_latency_ms": float(np.percentile(latencies_arr, 50)),
        "p90_latency_ms": float(np.percentile(latencies_arr, 90)),
        "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
        "p99_latency_ms": float(np.percentile(latencies_arr, 99)),
        "min_latency_ms": float(np.min(latencies_arr)),
        "max_latency_ms": float(np.max(latencies_arr)),
        "latencies_raw": latencies_arr.tolist(),
    }


async def main():
    parser = argparse.ArgumentParser(description="Triton Async gRPC Benchmark Suite")
    parser.add_argument("--url", type=str, default="127.0.0.1:8001", help="Triton gRPC endpoint")
    parser.add_argument("--model", type=str, default="cyclotron_ensemble", help="Target model name")
    parser.add_argument("--requests", type=int, default=1000, help="Total requests per run")
    parser.add_argument(
        "--concurrency-list",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated list of worker concurrencies",
    )
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file path")
    args = parser.parse_args()

    client = grpcclient.InferenceServerClient(url=args.url)

    if not await client.is_model_ready(args.model):
        raise RuntimeError(f"Model '{args.model}' is not ready on Triton gRPC server at {args.url}")

    concurrencies = [int(c.strip()) for c in args.concurrency_list.split(",")]
    all_results = {
        "model_name": args.model,
        "timestamp": time.time(),
        "runs": [],
    }

    print(f"Starting SLA Validation Suite for model '{args.model}' on {args.url}")
    print("=" * 70)

    for c in concurrencies:
        print(f"Running concurrency={c} ({args.requests} requests)...")
        res = await benchmark_concurrency(client, args.model, args.requests, c)
        all_results["runs"].append(res)

        print(
            f" -> Throughput: {res['throughput_fps']:.2f} infer/sec | "
            f"P50: {res['p50_latency_ms']:.2f}ms | "
            f"P90: {res['p90_latency_ms']:.2f}ms | "
            f"P99: {res['p99_latency_ms']:.2f}ms"
        )

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 70)
    print(f"Benchmark run complete. Results exported to '{args.output}'.")


if __name__ == "__main__":
    asyncio.run(main())