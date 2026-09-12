import asyncio
import cv2
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient

TRITON_URL = "localhost:8001"

async def process_stream(source=0):
    # Initialize Triton Async gRPC Client
    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    
    # Open Video Source (0 for V4L2 webcam, string URL for RTSP stream)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Unable to open video source: {source}")
        return

    print(f"--- Starting Live Stream Pipeline on Source: {source} ---")
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] End of stream or frame drop.")
                break

            frame_count += 1
            h, w, _ = frame.shape

            # Extract 12D feature vector from frame metadata and spatial geometry
            vec_12d = np.array([[
                0.0, 0.0, float(w), float(h),
                float(w / h), float(h / 2.0), float(w / 2.0),
                float(w * h), 0.5, 0.1, 0.0, 1.0
            ]], dtype=np.float32)

            # Construct gRPC input payload
            inputs = [grpcclient.InferInput("input_vector", vec_12d.shape, "FP32")]
            inputs[0].set_data_from_numpy(vec_12d)

            # Dispatch asynchronous inference to Triton
            response = await client.infer(model_name="ccvnn_hardswish_12d", inputs=inputs)
            output = response.as_numpy("output_prediction")

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"[Stream Frame #{frame_count:04d}] FPS: {fps:.2f} | Latest Output: {output[0][0]:.4f}")

            # Cap test execution at 150 frames (~5 seconds at 30 FPS)
            if frame_count >= 150:
                break

    finally:
        cap.release()
        await client.close()
        print(f"--- Stream Ingestion Finished: Processed {frame_count} frames ---")

if __name__ == "__main__":
    # Test with synthetic GStreamer test pattern (zero external camera dependency)
    gst_pipeline = "videotestsrc pattern=ball ! video/x-raw,width=640,height=480,framerate=30/1 ! appsink drop=true"
    asyncio.run(process_stream(gst_pipeline))