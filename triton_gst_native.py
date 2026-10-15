import asyncio
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

TRITON_URL = "localhost:8001"

async def run_gst_triton_pipeline(total_frames=150):
    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    
    # Construct GStreamer pipeline string
    pipeline_str = (
        "videotestsrc pattern=ball ! "
        "video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! "
        "appsink name=sink emit-signals=true max-buffers=1 drop=true"
    )
    
    pipeline = Gst.parse_launch(pipeline_str)
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)
    
    print(f"--- Native GStreamer Pipeline Active ({total_frames} frames) ---")
    start_time = time.time()

    try:
        for frame_count in range(1, total_frames + 1):
            # Pull frame buffer from GStreamer appsink
            sample = sink.emit("pull-sample")
            if not sample:
                continue

            buf = sample.get_buffer()
            caps = sample.get_caps()
            height = caps.get_structure(0).get_value("height")
            width = caps.get_structure(0).get_value("width")

            # Extract 12D feature vector matching ccvnn_hardswish_12d
            vec_12d = np.array([[
                0.0, 0.0, float(width), float(height),
                float(width / height), float(height / 2.0), float(width / 2.0),
                float(width * height), 0.5, 0.1, 0.0, 1.0
            ]], dtype=np.float32)

            # Dispatch asynchronous gRPC inference to Triton
            inputs = [grpcclient.InferInput("input_vector", vec_12d.shape, "FP32")]
            inputs[0].set_data_from_numpy(vec_12d)

            response = await client.infer(model_name="ccvnn_hardswish_12d", inputs=inputs)
            output = response.as_numpy("output_prediction")

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"[GStreamer Frame #{frame_count:04d}] Throughput: {fps:.2f} FPS | Prediction: {output[0][0]:.4f}")

    finally:
        pipeline.set_state(Gst.State.NULL)
        await client.close()
        print(f"--- Pipeline Finished: Processed {total_frames} frames ---")

if __name__ == "__main__":
    asyncio.run(run_gst_triton_pipeline())