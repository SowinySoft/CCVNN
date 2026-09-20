import os
import asyncio
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient

TRITON_URL = os.getenv("TRITON_URL", "localhost:8001")
TARGET_MODEL_NAME = os.getenv("MODEL_NAME", "ccvnn_v14_model_b")

def generate_v14_payload(batch_size: int = 1) -> np.ndarray:
    """Generates a synthetic batch of 14-element FP32 feature vectors."""
    # Features [0..11]: Spatial metadata | Feature [12]: normRotation | Feature [13]: saturationIndex
    return np.random.uniform(low=0.0, high=1.0, size=(batch_size, 14)).astype(np.float32)

async def process_simulated_stream(total_frames=150, fps_target=30):
    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    print(f"--- Starting CCVNN V14 Stream Simulator ({total_frames} frames @ {fps_target} FPS) ---")
    print(f"--- Target: {TRITON_URL} | Model: {TARGET_MODEL_NAME} ---")
    
    start_time = time.time()
    frame_delay = 1.0 / fps_target

    try:
        for frame_count in range(1, total_frames + 1):
            vec_14d = generate_v14_payload(batch_size=1)

            inputs = [grpcclient.InferInput("input_vector", vec_14d.shape, "FP32")]
            inputs[0].set_data_from_numpy(vec_14d)

            response = await client.infer(model_name=TARGET_MODEL_NAME, inputs=inputs)
            output = response.as_numpy("output_prediction")

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"[Sim Frame #{frame_count:04d}] FPS: {actual_fps:.2f} | Latest Output: {output[0][0]:.4f}")

            await asyncio.sleep(frame_delay)

    except Exception as e:
        print(f"[!] Stream simulator error: {e}")
    finally:
        await client.close()
        print(f"--- Stream Ingestion Finished: Processed {total_frames} frames ---")

if __name__ == "__main__":
    asyncio.run(process_simulated_stream())