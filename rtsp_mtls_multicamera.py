import asyncio
import time
import numpy as np
import tritonclient.grpc.aio as grpcclient
import camera_config

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

TRITON_URL = "localhost:8001"
CA_CERT = "certs/ca.crt"
CLIENT_CERT = "certs/client.crt"
CLIENT_KEY = "certs/client.key"

CAMERA_STREAMS = [
    {
        "id": "Cam_01_Hikvision_3K",
        "url": camera_config.CameraProfiles.get_hikvision_rtsp("192.168.1.100", "admin", "pass123", channel=1, stream="main"),
        "model": "ccvnn_hardswish_12d",
        "dim": 12
    },
    {
        "id": "Cam_03_UNV_5MP",
        "url": camera_config.CameraProfiles.get_unv_rtsp("192.168.1.105", "admin", "pass123", stream="main"),
        "model": "ccvnn_hardswish_12d",
        "dim": 12
    }
]

async def mtls_pipeline_worker(camera_info):
    cam_id = camera_info["id"]
    rtsp_url = camera_info["url"]
    model_name = camera_info["model"]
    input_dim = camera_info["dim"]

    pipeline_str = f"""
        rtspsrc location={rtsp_url} latency=100 drop-on-latency=true !
        rtph264depay !
        h264parse !
        nvv4l2decoder !
        nvvideoconvert !
        video/x-raw(memory:NVMM), format=NV12 !
        nvvideoconvert !
        video/x-raw, format=BGR !
        appsink name=sink_{cam_id} emit-signals=True max-buffers=1 drop=True
    """

    print(f"[mTLS START] Launching Secure Pipeline for {cam_id}...")
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name(f"sink_{cam_id}")

    if not appsink:
        print(f"[ERROR] Failed to attach appsink for {cam_id}")
        return

    pipeline.set_state(Gst.State.PLAYING)

    # Read mTLS Credentials
    with open(CA_CERT, 'rb') as f:
        root_certificates = f.read()
    with open(CLIENT_CERT, 'rb') as f:
        cert_chain = f.read()
    with open(CLIENT_KEY, 'rb') as f:
        private_key = f.read()

    # Create Secure gRPC Triton Client with mTLS
    client = grpcclient.InferenceServerClient(
        url=TRITON_URL,
        ssl=True,
        root_certificates=root_certificates,
        private_key=private_key,
        certificate_chain=cert_chain
    )

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            sample = appsink.emit("pull-sample")
            if not sample:
                await asyncio.sleep(0.001)
                continue

            buffer = sample.get_buffer()
            caps = sample.get_caps()
            height = caps.get_structure(0).get_value("height")
            width = caps.get_structure(0).get_value("width")

            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                continue

            data = np.random.randn(1, input_dim).astype(np.float32)
            buffer.unmap(map_info)

            inputs = [grpcclient.InferInput("input_vector", data.shape, "FP32")]
            inputs[0].set_data_from_numpy(data)

            t0 = time.perf_counter()
            response = await client.infer(model_name=model_name, inputs=inputs)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
                print(f"[{cam_id} | SECURE mTLS] Res: {width}x{height} | Rate: {fps:.1f} FPS | Latency: {latency_ms:.2f} ms")

            await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        await client.close()
        print(f"[mTLS END] Pipeline {cam_id} stopped.")

async def main():
    print("--- Starting Step 4.2 Mutual TLS (mTLS) Ingestion Pipeline ---")
    tasks = [asyncio.create_task(mtls_pipeline_worker(cam)) for cam in CAMERA_STREAMS]
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping mTLS ingestion manager...")
        for t in tasks:
            t.cancel()

if __name__ == "__main__":
    asyncio.run(main())
