import asyncio
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException

TRITON_URL = "localhost:8001"
MODEL_NAME = "ccvnn_v20_backbone"

async def send_async_request(client, req_id, input_data, input_name, output_name):
    """Sends a single asynchronous inference request and measures latency."""
    inputs = [grpcclient.InferInput(input_name, input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    outputs = [grpcclient.InferRequestedOutput(output_name)]

    start_time = time.perf_counter()
    response = await client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    _ = response.as_numpy(output_name)
    return latency_ms

async def main():
    # Initialize async gRPC client
    client = grpcclient.InferenceServerClient(url=TRITON_URL)

    try:
        if not await client.is_server_ready():
            print(f"[ERROR] Triton server at {TRITON_URL} is not ready.")
            return
        if not await client.is_model_ready(MODEL_NAME):
            print(f"[ERROR] Model '{MODEL_NAME}' is not loaded/ready.")
            return
    except Exception as e:
        print(f"[ERROR] Could not connect to Triton: {e}")
        return

    # Configuration
    INPUT_NAME = "input"      # Update to match config.pbtxt
    OUTPUT_NAME = "output"    # Update to match config.pbtxt
    CONCURRENCY = 20          # Number of concurrent async requests
    BATCH_SIZE = 8
    VECTOR_DIM = 512

    # Generate test payload
    input_data = np.random.randn(BATCH_SIZE, VECTOR_DIM).astype(np.float32)

    print("========================================================")
    print(f" Starting Async Concurrency Stress Test on '{MODEL_NAME}'")
    print(f" Parallel Tasks: {CONCURRENCY} | Batch Size: {BATCH_SIZE} | Tensor Shape: ({BATCH_SIZE}, {VECTOR_DIM})")
    print("========================================================\n")

    # Dispatch tasks concurrently
    tasks = [
        send_async_request(client, i, input_data, INPUT_NAME, OUTPUT_NAME)
        for i in range(CONCURRENCY)
    ]

    wall_start = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_duration = time.perf_counter() - wall_start

    # Process metrics
    latencies = []
    failures = 0
    for r in results:
        if isinstance(r, Exception):
            failures += 1
        else:
            latencies.append(r)

    if latencies:
        lat_arr = np.array(latencies)
        total_inferences = len(lat_arr) * BATCH_SIZE

        print("---------------- Benchmark Summary ----------------")
        print(f" Successful Requests : {len(lat_arr)} / {CONCURRENCY}")
        print(f" Failed Requests     : {failures}")
        print(f" Total Inferences    : {total_inferences}")
        print(f" Total Elapsed Time  : {wall_duration:.3f} s")
        print(f" System Throughput   : {total_inferences / wall_duration:.2f} infer/sec")
        print(f" Avg Client Latency  : {np.mean(lat_arr):.2f} ms")
        print(f" p50 Latency         : {np.percentile(lat_arr, 50):.2f} ms")
        print(f" p95 Latency         : {np.percentile(lat_arr, 95):.2f} ms")
        print(f" p99 Latency         : {np.percentile(lat_arr, 99):.2f} ms")
        print("---------------------------------------------------")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())