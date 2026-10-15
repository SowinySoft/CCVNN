import os
import asyncio
import cv2
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient

TRITON_URL = os.getenv("TRITON_URL", "localhost:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "ccvnn_v14_model_b")

async def process_stream(source=0):
    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[!] Unable to open video source: {source}")
        return

    print(f"--- Starting CCVNN V14 Live Stream Pipeline on Source: {source} ---")
    print(f"--- Triton Target: {TRITON_URL} | Model: {MODEL_NAME} ---")
    
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] End of stream or frame drop.")
                break

            frame_count += 1
            h, w, _ = frame.shape

            # Extract 14D feature vector matching V14 architecture spec
            vec_14d = np.array([[
                0.0, 0.0, float(w), float(h),
                float(w / h), float(h / 2.0), float(w / 2.0),
                float(w * h), 0.5, 0.1, 0.0, 1.0,
                0.0,  # normRotation (Feature 13)
                0.5   # saturationIndex (Feature 14)
            ]], dtype=np.float32)

            inputs = [grpcclient.InferInput("input_vector", vec_14d.shape, "FP32")]
            inputs[0].set_data_from_numpy(vec_14d)

            response = await client.infer(model_name=MODEL_NAME, inputs=inputs)
            output = response.as_numpy("output_prediction")

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"[Stream Frame #{frame_count:04d}] FPS: {fps:.2f} | V14 Output: {output[0][0]:.4f}")

            if frame_count >= 150:
                break

    except Exception as e:
        print(f"[!] Pipeline processing error: {e}")
    finally:
        cap.release()
        await client.close()
        print(f"--- Stream Ingestion Finished: Processed {frame_count} frames ---")

if __name__ == "__main__":
    gst_pipeline = "videotestsrc pattern=ball ! video/x-raw,width=640,height=480,framerate=30/1 ! appsink drop=true"
    asyncio.run(process_stream(gst_pipeline))