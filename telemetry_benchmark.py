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
    latency_ms = (time.perf_counter() - start) * 1000.0
    return latency_ms

async def run_telemetry_benchmark(num_requests=100, concurrency=10):
    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    print(f"--- Starting End-to-End Latency Telemetry ({num_requests} total requests) ---")
    
    latencies = []
    start_total = time.perf_counter()

    for i in range(0, num_requests, concurrency):
        tasks = []
        for j in range(concurrency):
            # Alternate between hardswish_12d and relu6_9d
            if (i + j) % 2 == 0:
                tasks.append(send_request(client, "ccvnn_hardswish_12d", 12))
            else:
                tasks.append(send_request(client, "ccvnn_relu6_9d", 9))
        
        batch_latencies = await asyncio.gather(*tasks)
        latencies.extend(batch_latencies)

    total_time = time.perf_counter() - start_total
    rps = num_requests / total_time

    # Calculate percentiles
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print("\n================ TELEMETRY SUMMARY REPORT ================")
    print(f" Total Requests Sent  : {num_requests}")
    print(f" Total Execution Time : {total_time * 1000.0:.2f} ms")
    print(f" System Throughput    : {rps:.2f} requests/sec")
    print("----------------------------------------------------------")
    print(f" Min Latency          : {np.min(latencies):.2f} ms")
    print(f" Mean Latency         : {np.mean(latencies):.2f} ms")
    print(f" p50 (Median) Latency : {p50:.2f} ms")
    print(f" p95 Latency          : {p95:.2f} ms")
    print(f" p99 Latency          : {p99:.2f} ms")
    print(f" Max Latency          : {np.max(latencies):.2f} ms")
    print("==========================================================\n")

    await client.close()

if __name__ == "__main__":
    asyncio.run(run_telemetry_benchmark(num_requests=100, concurrency=10))
