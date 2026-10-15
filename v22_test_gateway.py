import argparse
import asyncio
import os
import time
import httpx
import numpy as np


async def send_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    gateway_url: str,
    payload: dict,
):
    async with semaphore:
        try:
            start_time = time.perf_counter()
            response = await client.post(gateway_url, json=payload)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return response.status_code, latency_ms
        except httpx.ConnectTimeout:
            return "CONNECT_TIMEOUT", 0.0
        except httpx.ReadTimeout:
            return "READ_TIMEOUT", 0.0
        except httpx.HTTPError as exc:
            return f"HTTP_ERROR_{type(exc).__name__}", 0.0


async def benchmark_model(
    gateway_base: str,
    model_name: str,
    concurrency: int,
    total_requests: int,
    override_dim: int | None = None,
):
    gateway_url = f"{gateway_base.rstrip('/')}/v2/models/{model_name}/infer"
    health_url = f"{gateway_base.rstrip('/')}/health/live"

    # Select feature dimension based on model structure if not explicitly overridden
    if override_dim is not None:
        feature_dim = override_dim
    elif "yolo" in model_name.lower():
        feature_dim = 3 * 640 * 640  # 1,228,800
    else:
        feature_dim = 4 * 64 * 64    # 16,384

    # Pre-allocate payload based on feature-dim to prevent per-request memory overhead
    payload = {"data": [0.1] * feature_dim}

    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(30.0, connect=5.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        try:
            health = await client.get(health_url)
            if health.status_code != 200:
                print(f"[ERROR] Health check failed for model '{model_name}' (Status: {health.status_code})")
                return None
        except Exception as exc:
            print(f"[ERROR] Gateway unreachable at {health_url}: {exc}")
            return None

        print(f"[BENCHMARK] Testing model: '{model_name}' ({total_requests} reqs, {concurrency} workers, dim {feature_dim})...")
        start_total = time.perf_counter()

        tasks = [
            send_request(client, semaphore, gateway_url, payload)
            for _ in range(total_requests)
        ]
        results = await asyncio.gather(*tasks)

        total_elapsed = time.perf_counter() - start_total

    status_counts = {}
    latencies = []

    for status_code, lat in results:
        status_counts[status_code] = status_counts.get(status_code, 0) + 1
        if status_code == 200:
            latencies.append(lat)

    throughput = total_requests / total_elapsed if total_elapsed > 0 else 0.0

    return {
        "model": model_name,
        "total_time": f"{total_elapsed:.2f} s",
        "throughput": f"{throughput:.2f} req/s",
        "status_breakdown": str(status_counts),
        "p50": f"{np.percentile(latencies, 50):.2f} ms" if latencies else "N/A",
        "p95": f"{np.percentile(latencies, 95):.2f} ms" if latencies else "N/A",
        "p99": f"{np.percentile(latencies, 99):.2f} ms" if latencies else "N/A",
    }


async def main():
    parser = argparse.ArgumentParser(description="Multi-Model v22 Gateway Load Benchmark")
    parser.add_argument(
        "--models",
        type=str,
        default=os.getenv("TRITON_MODELS", "ccvnn"),
        help="Comma-separated list of target model names deployed on Triton",
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        default=os.getenv("GATEWAY_URL", "http://127.0.0.1:8000"),
        help="Base FastAPI gateway URL",
    )
    parser.add_argument("--concurrency", type=int, default=32, help="Concurrent HTTP workers")
    parser.add_argument("--total-requests", type=int, default=320, help="Total requests to send per model")
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=None,
        help="Override feature vector payload dimension for all models",
    )

    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    all_results = []

    for model in model_list:
        res = await benchmark_model(
            gateway_base=args.gateway_url,
            model_name=model,
            concurrency=args.concurrency,
            total_requests=args.total_requests,
            override_dim=args.feature_dim,
        )
        if res:
            all_results.append(res)

    if not all_results:
        print("\n[ERROR] No benchmark results collected.")
        return

    # Formatted Benchmark Summary Table
    print("\n=================================================================================================================")
    print("                                            ALL MODELS BENCHMARK REPORT                                          ")
    print("=================================================================================================================")
    print(f"| {'Target Model':<22} | {'Total Time':<10} | {'Throughput':<12} | {'Status Breakdown':<22} | {'P50 Latency':<12} | {'P95 Latency':<12} | {'P99 Latency':<12} |")
    print("|" + "|".join(["-"*24, "-"*12, "-"*14, "-"*24, "-"*14, "-"*14, "-"*14]) + "|")

    for r in all_results:
        print(
            f"| {r['model']:<22} | {r['total_time']:<10} | {r['throughput']:<12} | {r['status_breakdown']:<22} | {r['p50']:<12} | {r['p95']:<12} | {r['p99']:<12} |"
        )
    print("=================================================================================================================\n")


if __name__ == "__main__":
    asyncio.run(main())