import time
import os
import psutil
import numpy as np
import onnxruntime as ort

def profile_v17_inference(iterations=10000):
    process = psutil.Process(os.getpid())
    model_path = "ccvnn_v17_backbone_int8.onnx"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found.")
        return

    # Initialize ONNX session
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # Generate dummy batch of 17-element vectors (float32)
    dummy_input = np.random.randn(1, 17).astype(np.float32)

    # Initial resource baseline
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    cpu_start_time = time.process_time()
    wall_start_time = time.perf_counter()

    # High-frequency inference loop
    for _ in range(iterations):
        _ = session.run(None, {input_name: dummy_input})

    wall_end_time = time.perf_counter()
    cpu_end_time = time.process_time()
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB

    # Calculations
    total_wall_time = wall_end_time - wall_start_time
    total_cpu_time = cpu_end_time - cpu_start_time
    avg_latency_us = (total_wall_time / iterations) * 1e6
    throughput_fps = iterations / total_wall_time
    cpu_utilization = (total_cpu_time / total_wall_time) * 100

    print("\n" + "=" * 55)
    print("📊 CCVNN V17 HIGH-FREQUENCY INFERENCE PROFILE REPORT")
    print("=" * 55)
    print(f"Total Inferences Evaluated : {iterations:,}")
    print(f"Total Wall-Clock Time     : {total_wall_time:.4f} sec")
    print(f"Avg Latency per Inference : {avg_latency_us:.3f} µs")
    print(f"Inference Throughput       : {throughput_fps:,.2f} FPS")
    print(f"Effective CPU Utilization  : {cpu_utilization:.2f} %")
    print(f"Initial Memory Footprint  : {mem_before:.2f} MB")
    print(f"Peak Memory Footprint     : {mem_after:.2f} MB")
    print(f"Memory Delta              : {mem_after - mem_before:.2f} MB")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    profile_v17_inference()
