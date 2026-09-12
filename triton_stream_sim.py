import asyncio
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient

TRITON_URL = "localhost:8001"

async def process_simulated_stream(total_frames=150, fps_target=30):
    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    print(f"--- Starting Zero-Dependency Stream Simulator ({total_frames} frames) ---")
    
    start_time = time.time()
    frame_delay = 1.0 / fps_target

    try:
        for frame_count in range(1, total_frames + 1):
            # Generate synthetic 12D feature vector matching 640x480 frame dimensions
            vec_12d = np.array([[
                0.0, 0.0, 640.0, 480.0,
                1.333, 240.0, 320.0,
                307200.0, 0.5, 0.1, 0.0, 1.0
            ]], dtype=np.float32)

            inputs = [grpcclient.InferInput("input_vector", vec_12d.shape, "FP32")]
            inputs[0].set_data_from_numpy(vec_12d)

            # Dispatch asynchronous inference over gRPC
            response = await client.infer(model_name="ccvnn_hardswish_12d", inputs=inputs)
            output = response.as_numpy("output_prediction")

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"[Stream Frame #{frame_count:04d}] FPS: {actual_fps:.2f} | Latest Output: {output[0][0]:.4f}")

            await asyncio.sleep(frame_delay)

    finally:
        await client.close()
        print(f"--- Stream Ingestion Finished: Processed {total_frames} frames ---")

if __name__ == "__main__":
    asyncio.run(process_simulated_stream())