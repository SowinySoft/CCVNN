import argparse
import asyncio
import sys
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

def parse_args():
    parser = argparse.ArgumentParser(description="Live RTSP Stream Ingestion to Triton Inference Server")
    parser.add_argument("--rtsp-url", type=str, required=True, help="RTSP stream URL (e.g., rtsp://admin:pass@192.168.1.100:554/Streaming/Channels/101)")
    parser.add_argument("--triton-url", type=str, default="localhost:8001", help="Triton gRPC server address")
    parser.add_argument("--model-name", type=str, default="ccvnn_hardswish_12d", help="Target model on Triton")
    parser.add_argument("--input-dim", type=int, default=12, help="Model input feature vector size")
    return parser.parse_args()

async def process_rtsp_stream(rtsp_url, triton_url, model_name, input_dim):
    # Construct GStreamer RTSP Pipeline
    pipeline_str = f"""
        rtspsrc location={rtsp_url} latency=100 drop-on-latency=true !
        rtph264depay !
        h264parse !
        avdec_h264 !
        videoconvert !
        video/x-raw, format=BGR !
        appsink name=sink emit-signals=True max-buffers=1 drop=True
    """
    
    print(f"[INFO] Initializing RTSP Pipeline for URL: {rtsp_url}")
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name("sink")

    if not appsink:
        print("[ERROR] Could not retrieve appsink element from pipeline.")
        return

    pipeline.set_state(Gst.State.PLAYING)
    client = grpcclient.InferenceServerClient(url=triton_url)
    
    print(f"[INFO] Streaming live frames to Triton server at '{triton_url}' using model '{model_name}'...")
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Pull frame buffer from RTSP stream (timeout 2 seconds)
            sample = appsink.emit("pull-sample")
            if not sample:
                await asyncio.sleep(0.001)
                continue

            buffer = sample.get_buffer()
            caps = sample.get_caps()
            height = caps.get_structure(0).get_value("height")
            width = caps.get_structure(0).get_value("width")

            # Extract raw frame array
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                continue

            # Generate synthetic 12D/9D feature embedding or preprocessed frame representation
            dummy_features = np.random.randn(1, input_dim).astype(np.float32)
            buffer.unmap(map_info)

            # Fire Async Inference Request to Triton
            inputs = [grpcclient.InferInput("input_vector", dummy_features.shape, "FP32")]
            inputs[0].set_data_from_numpy(dummy_features)
            
            t0 = time.perf_counter()
            response = await client.infer(model_name=model_name, inputs=inputs)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"[LIVE STREAM] Processed Frame {frame_count:05d} ({width}x{height}) | Stream Rate: {fps:.2f} FPS | Triton Latency: {latency_ms:.2f} ms")

            await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] Live stream benchmark stopped by user.")
    finally:
        pipeline.set_state(Gst.State.NULL)
        await client.close()
        print("[INFO] RTSP pipeline shut down successfully.")

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(process_rtsp_stream(args.rtsp_url, args.triton_url, args.model_name, args.input_dim))
