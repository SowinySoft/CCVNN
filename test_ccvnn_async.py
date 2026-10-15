import asyncio
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient


async def warm_up_model(client: grpcclient.InferenceServerClient, passes: int = 5):
    """Pre-warms the Triton model and C++ execution handles."""
    dummy_input = np.random.randn(1, 17).astype(np.float32)
    inputs = [grpcclient.InferInput("v_input_17", dummy_input.shape, "FP32")]
    inputs[0].set_data_from_numpy(dummy_input)
    outputs = [grpcclient.InferRequestedOutput("hazard_prob")]

    for _ in range(passes):
        await client.infer(
            model_name="ccvnn_v17_backbone", inputs=inputs, outputs=outputs
        )


async def benchmark_pipelined(
    client: grpcclient.InferenceServerClient,
    total_requests: int = 1000,
    concurrency: int = 16,
):
    semaphore = asyncio.Semaphore(concurrency)
    dummy_input = np.random.randn(1, 17).astype(np.float32)

    async def worker(req_id: int):
        async with semaphore:
            inputs = [
                grpcclient.InferInput("v_input_17", dummy_input.shape, "FP32")
            ]
            inputs[0].set_data_from_numpy(dummy_input)
            outputs = [grpcclient.InferRequestedOutput("hazard_prob")]

            start = time.perf_counter()
            _ = await client.infer(
                model_name="ccvnn_v17_backbone",
                inputs=inputs,
                outputs=outputs,
                request_id=str(req_id),
            )
            return (time.perf_counter() - start) * 1000.0

    print(
        f"[>] Running continuous pipeline: {total_requests} requests across {concurrency} workers..."
    )
    start_all = time.perf_counter()
    latencies = await asyncio.gather(*[worker(i) for i in range(total_requests)])
    total_time = time.perf_counter() - start_all

    print(
        f"\n[+] Total Pipeline Throughput: {total_requests / total_time:.2f} infer/sec"
    )
    print(f"[+] Avg Pipelined Latency: {np.mean(latencies):.2f} ms")
    print(f"[+] p90 Pipelined Latency: {np.percentile(latencies, 90):.2f} ms")


async def main():
    async with grpcclient.InferenceServerClient(url="localhost:8001") as client:
        await warm_up_model(client)
        await benchmark_pipelined(
            client, total_requests=1000, concurrency=16
        )


if __name__ == "__main__":
    asyncio.run(main())