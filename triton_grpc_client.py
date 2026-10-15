import asyncio
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient

TRITON_GRPC_URL = "127.0.0.1:8001"

# Target tensor specifications matching config.pbtxt
MODEL_CONFIGS = {
    "ccvnn_hardswish_12d": {
        "input": "input_vector",
        "output": "output_prediction",
        "dim": 12,
    },
    "ccvnn_relu6_9d": {
        "input": "input_vector",
        "output": "output_coords",
        "dim": 9,
    },
}


async def send_grpc_inference(
    client: grpcclient.InferenceServerClient, model_name: str, req_id: int
):
    cfg = MODEL_CONFIGS[model_name]

    # 1. Prepare FP32 tensor [1, input_dim]
    dummy_input = np.random.randn(1, cfg["dim"]).astype(np.float32)

    # 2. Build Triton gRPC InferInput & InferRequestedOutput
    inputs = [grpcclient.InferInput(cfg["input"], dummy_input.shape, "FP32")]
    inputs[0].set_data_from_numpy(dummy_input)

    outputs = [grpcclient.InferRequestedOutput(cfg["output"])]

    # 3. Measure IPC round-trip latency + serialization overhead
    t_start = time.perf_counter()
    response = await client.infer(
        model_name=model_name, inputs=inputs, outputs=outputs
    )
    latency_ms = (time.perf_counter() - t_start) * 1000

    # 4. Extract output array
    result_array = response.as_numpy(cfg["output"])
    output_repr = (
        np.array2string(result_array.flatten(), precision=4, suppress_small=True)
        if result_array is not None
        else "None"
    )

    print(
        f"[gRPC #{req_id:02d}] {model_name:20s} | Latency: {latency_ms:6.2f}ms | Output: {output_repr}"
    )
    return result_array


async def main():
    print("--- Starting Async gRPC Inference Pipeline (Triton) ---")

    async with grpcclient.InferenceServerClient(url=TRITON_GRPC_URL) as client:
        if not await client.is_server_ready():
            print("ERROR: Triton gRPC server is not ready at", TRITON_GRPC_URL)
            return

        print("Triton gRPC server status: READY\n")

        # Dispatch 10 concurrent requests across both models
        tasks = []
        for i in range(10):
            target_model = (
                "ccvnn_relu6_9d" if i % 2 == 0 else "ccvnn_hardswish_12d"
            )
            tasks.append(send_grpc_inference(client, target_model, i + 1))

        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time_ms = (time.perf_counter() - t0) * 1000

        successful = sum(1 for r in results if r is not None)
        print("------------------------------------------------------------------")
        print(
            f"Executed {len(tasks)} gRPC requests ({successful} succeeded) in {total_time_ms:.2f}ms"
        )


if __name__ == "__main__":
    asyncio.run(main())