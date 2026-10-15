import asyncio
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient

TRITON_URL = "localhost:8001"
MODEL_NAME = "ccvnn_v14_model_b"
FEATURE_DIM = 14

# Define traffic stages: (duration_in_seconds, worker_concurrency)
TRAFFIC_STAGES = [
    (20, 2),   # Low load baseline
    (30, 10),  # Moderate traffic
    (30, 25),  # Heavy load
    (15, 60),  # Peak burst (tests dynamic batching under saturation)
    (15, 5),   # Step down / recovery
]

async def send_infer_request(client, input_data):
    triton_input = grpcclient.InferInput("input_vector", [1, FEATURE_DIM], "FP32")
    triton_input.set_data_from_numpy(input_data)
    triton_output = grpcclient.InferRequestedOutput("output_prediction")

    try:
        await client.infer(model_name=MODEL_NAME, inputs=[triton_input], outputs=[triton_output])
        return True
    except Exception:
        return False

async def traffic_worker(client, stop_event, stats):
    dummy_input = np.random.randn(1, FEATURE_DIM).astype(np.float32)
    while not stop_event.is_set():
        success = await send_infer_request(client, dummy_input)
        if success:
            stats["success"] += 1
        else:
            stats["failure"] += 1
        await asyncio.sleep(0.0001)  # Yield execution thread

async def main():
    client = grpcclient.InferenceServerClient(url=TRITON_URL)

    if not await client.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Model '{MODEL_NAME}' is not ready on Triton at {TRITON_URL}.")

    print(f"=== Traffic Generator Active for '{MODEL_NAME}' ===")
    print("Press Ctrl+C to stop traffic generation.\n")

    cycle_count = 1
    try:
        while True:
            print(f"--- Starting Load Cycle #{cycle_count} ---")
            for duration, concurrency in TRAFFIC_STAGES:
                print(f"[Stage] Target Concurrency: {concurrency:>2} workers | Duration: {duration}s")
                stop_event = asyncio.Event()
                stats = {"success": 0, "failure": 0}

                # Spawn concurrent worker tasks
                tasks = [
                    asyncio.create_task(traffic_worker(client, stop_event, stats))
                    for _ in range(concurrency)
                ]

                start_time = time.perf_counter()
                await asyncio.sleep(duration)

                # Signal workers to stop and await completion
                stop_event.set()
                await asyncio.gather(*tasks, return_exceptions=True)
                elapsed = time.perf_counter() - start_time

                rps = stats["success"] / elapsed
                print(f"        Result: {stats['success']} successful requests ({rps:.1f} req/sec), {stats['failure']} failures")

            cycle_count += 1
    except asyncio.CancelledError:
        pass
    finally:
        await client.close()
        print("\nTraffic generator shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTraffic generator stopped by user.")