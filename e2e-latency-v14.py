import os
import time
import json
import numpy as np
import tritonclient.http as httpclient

TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "ccvnn_v14_model_b")
NUM_REQUESTS = int(os.getenv("NUM_REQUESTS", "1000"))
BATCH_SIZE = 1
FEATURE_DIM = 14

def run_v14_e2e_benchmark():
    try:
        client = httpclient.InferenceServerClient(url=TRITON_URL)
        if not client.is_model_ready(MODEL_NAME):
            raise RuntimeError(f"Model '{MODEL_NAME}' is not ready on Triton server at {TRITON_URL}.")
    except Exception as e:
        print(f"[!] Triton Client Connection Error: {e}")
        return

    # Prepare payload once to isolate network/inference overhead
    dummy_input = np.random.randn(BATCH_SIZE, FEATURE_DIM).astype(np.float32)
    triton_input = httpclient.InferInput("input_vector", dummy_input.shape, "FP32")
    triton_input.set_data_from_numpy(dummy_input)
    triton_output = httpclient.InferRequestedOutput("output_prediction")

    # Warmup run
    print(f"[+] Executing 50 warmup requests for {MODEL_NAME}...")
    for _ in range(50):
        client.infer(model_name=MODEL_NAME, inputs=[triton_input], outputs=[triton_output])

    # Benchmark execution
    print(f"[+] Benchmarking {NUM_REQUESTS} requests against {TRITON_URL}...")
    latencies_ms = []
    benchmark_start_time = time.perf_counter()

    for _ in range(NUM_REQUESTS):
        t_start = time.perf_counter()
        client.infer(model_name=MODEL_NAME, inputs=[triton_input], outputs=[triton_output])
        t_end = time.perf_counter()
        latencies_ms.append((t_end - t_start) * 1000.0)

    total_duration_sec = time.perf_counter() - benchmark_start_time
    latencies = np.array(latencies_ms)
    throughput = (NUM_REQUESTS * BATCH_SIZE) / total_duration_sec

    results = {
        "model_name": MODEL_NAME,
        "total_requests": NUM_REQUESTS,
        "elapsed_sec": round(total_duration_sec, 4),
        "throughput_fps": round(throughput, 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 3),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 3),
        "p90_latency_ms": round(float(np.percentile(latencies, 90)), 3),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 3),
        "min_latency_ms": round(float(np.min(latencies)), 3),
        "max_latency_ms": round(float(np.max(latencies)), 3)
    }

    print("\n================ CCVNN V14 E2E RESULTS ================")
    print(f"Model Name:               {results['model_name']}")
    print(f"Total Requests Processed: {results['total_requests']}")
    print(f"Total Elapsed Time:       {results['elapsed_sec']} sec")
    print(f"Throughput:               {results['throughput_fps']} infer/sec")
    print(f"Mean Latency:             {results['mean_latency_ms']} ms")
    print(f"P50 Latency (Median):     {results['p50_latency_ms']} ms")
    print(f"P90 Latency:              {results['p90_latency_ms']} ms")
    print(f"P99 Latency:              {results['p99_latency_ms']} ms")
    print(f"Min / Max Latency:        {results['min_latency_ms']} ms / {results['max_latency_ms']} ms")
    print("======================================================")

    # Export to json for pipeline profiling
    os.makedirs("public", exist_ok=True)
    with open("public/e2e_results_v14.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_v14_e2e_benchmark()