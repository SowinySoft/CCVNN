import asyncio
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient

TRITON_URL = "localhost:8001"

async def send_request(client, model_name, input_dim):
    data = np.random.randn(1, input_dim).astype(np.float32)
    inputs = [grpcclient.InferInput("input_vector", data.shape, "FP32")]
    inputs[0].set_data_from_numpy(data)
    
    start = time.perf_counter()
    response = await client.infer(model_name=model_name, inputs=inputs)
    return (time.perf_counter() - start) * 1000.0

async def run_benchmark(num_requests=100, concurrency=10):
    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    print("--- Telemetry: ccvnn_hardswish_12d ONLY (100 requests) ---")
    
    latencies = []
    start_total = time.perf_counter()

    for i in range(0, num_requests, concurrency):
        tasks = [send_request(client, "ccvnn_hardswish_12d", 12) for _ in range(concurrency)]
        batch_latencies = await asyncio.gather(*tasks)
        latencies.extend(batch_latencies)

    total_time = time.perf_counter() - start_total
    rps = num_requests / total_time
    latencies = np.array(latencies)

    print("\n================ HARDSWISH_12D SUMMARY ================")
    print(f" Total Throughput : {rps:.2f} requests/sec")
    print(f" Mean Latency     : {np.mean(latencies):.2f} ms")
    print(f" p50 (Median)     : {np.percentile(latencies, 50):.2f} ms")
    print(f" p95 Latency      : {np.percentile(latencies, 95):.2f} ms")
    print(f" p99 Latency      : {np.percentile(latencies, 99):.2f} ms")
    print("=======================================================\n")

    await client.close()

if __name__ == "__main__":
    asyncio.run(run_benchmark())
