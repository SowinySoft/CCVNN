import asyncio
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient


async def infer_model(
    client, model_name, input_name, output_name, input_shape, data_type="FP32"
):
    start_time = time.perf_counter()

    # Generate random test vector matching model specs
    input_data = np.random.randn(*input_shape).astype(np.float32)

    inputs = [grpcclient.InferInput(input_name, input_data.shape, data_type)]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [grpcclient.InferRequestedOutput(output_name)]

    response = await client.infer(
        model_name=model_name, inputs=inputs, outputs=outputs
    )
    result_np = response.as_numpy(output_name)

    elapsed = (time.perf_counter() - start_time) * 1000
    return model_name, result_np, elapsed


async def main():
    # Initialize asynchronous gRPC client
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Define concurrent tasks for both models
    tasks = [
        infer_model(
            client,
            "ccvnn_hardswish_12d",
            "input_vector",
            "output_prediction",
            (1, 12),
        ),
        infer_model(
            client,
            "ccvnn_relu6_9d",
            "input_vector",
            "output_coords",
            (1, 9),
        ),
    ]

    print("Sending concurrent async gRPC inference requests...")
    results = await asyncio.gather(*tasks)

    for model_name, res, latency in results:
        print(f"Model: {model_name}")
        print(f"  Result Shape: {res.shape if res is not None else 'None'}")
        print(f"  Result Values: {res}")
        print(f"  Round-trip Latency: {latency:.2f} ms\n")


if __name__ == "__main__":
    asyncio.run(main())