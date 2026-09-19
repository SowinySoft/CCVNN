import asyncio
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient

TRITON_URL = "localhost:8001"
MODEL_NAME = "ccvnn_v14_model_b"
CONCURRENCY = 8          # Number of concurrent client tasks
TOTAL_REQUESTS = 1000    # Total requests distributed across workers
BATCH_SIZE = 1
FEATURE_DIM = 14

async def worker(client, reqs_per_worker, latencies, triton_input, triton_output):
    for _ in range(reqs_per_worker):
        t_start = time.perf_counter()
        await client.infer(
            model_name=MODEL_NAME,
            inputs=[triton_input],
            outputs=[triton_output]
        )
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000.0)

async def main():
    client = grpcclient.InferenceServerClient(url=TRITON_URL)

    if not await client.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Model '{MODEL_NAME}' is not ready on Triton gRPC server.")

    # Prepare input payload matching [-1, 14] shape
    dummy_input = np.random.randn(BATCH_SIZE, FEATURE_DIM).astype(np.float32)
    triton_input = grpcclient.InferInput("input_vector", list(dummy_input.shape), "FP32")
    triton_input.set_data_from_numpy(dummy_input)
    triton_output = grpcclient.InferRequestedOutput("output_prediction")

    # Warmup runs
    print("Executing 50 warmup requests...")
    for _ in range(50):
        await client.infer(model_name=MODEL_NAME, inputs=[triton_input], outputs=[triton_output])

    print(f"Benchmarking {TOTAL_REQUESTS} requests across {CONCURRENCY} concurrent workers...")

    latencies = []
    reqs_per_worker = TOTAL_REQUESTS // CONCURRENCY

    tasks = [
        worker(client, reqs_per_worker, latencies, triton_input, triton_output)
        for _ in range(CONCURRENCY)
    ]

    t_start_total = time.perf_counter()
    await asyncio.gather(*tasks)
    t_end_total = time.perf_counter()

    total_duration = t_end_total - t_start_total
    actual_requests = reqs_per_worker * CONCURRENCY
    throughput = actual_requests / total_duration
    lat = np.array(latencies)

    print("\n================ ASYNC gRPC BENCHMARK RESULTS ================")
    print(f"Concurrency Level:       {CONCURRENCY} worker tasks")
    print(f"Total Requests Processed:{actual_requests}")
    print(f"Total Elapsed Time:      {total_duration:.4f} sec")
    print(f"Throughput:              {throughput:.2f} infer/sec")
    print(f"Mean Latency:            {np.mean(lat):.3f} ms")
    print(f"P50 Latency (Median):    {np.percentile(lat, 50):.3f} ms")
    print(f"P90 Latency:             {np.percentile(lat, 90):.3f} ms")
    print(f"P99 Latency:             {np.percentile(lat, 99):.3f} ms")
    print(f"Min / Max Latency:       {np.min(lat):.3f} ms / {np.max(lat):.3f} ms")
    print("==============================================================")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())