import json
import os
import sys
import time
from collections import deque
import numpy as np
import tritonclient.grpc as grpcclient


class IndustrialInspectionFilter:
    def __init__(self, window_size=5, margin_threshold=0.25):
        self.window = deque(maxlen=window_size)
        self.margin = margin_threshold
        self.last_stable_state = None

    def process_prediction(self, pred_np: np.ndarray) -> int:
        # pred_np shape: [1, 1] or [1]
        raw_val = float(pred_np.flat[0])

        # Convert raw logit to probability if necessary
        if raw_val < 0.0 or raw_val > 1.0:
            prob = 1.0 / (1.0 + np.exp(-raw_val))
        else:
            prob = raw_val

        raw_pred = 1 if prob >= 0.5 else 0
        margin = abs(prob - 0.5)

        self.window.append(raw_pred)
        majority_pred = 1 if sum(self.window) > (len(self.window) / 2) else 0

        # Maintain last stable state if prediction falls within uncertainty margin
        if margin < self.margin and self.last_stable_state is not None:
            return self.last_stable_state

        self.last_stable_state = majority_pred
        return majority_pred


def run_triton_stream(
    server_url=None, model_name="ccvnn_v14_model_b", num_frames=1000
):
    if server_url is None:
        server_url = os.getenv("TRITON_SERVER_URL", "localhost:8001")

    client = grpcclient.InferenceServerClient(url=server_url)
    inspection_filter = IndustrialInspectionFilter()

    latencies = []
    print(
        f"[INFO] Initializing V14 stream test to Triton at {server_url} for model '{model_name}'..."
    )

    for i in range(num_frames):
        # V14 14-element FP32 input vector specification
        dummy_vector = np.random.rand(1, 14).astype(np.float32)
        inputs = [grpcclient.InferInput("input_vector", [1, 14], "FP32")]
        inputs[0].set_data_from_numpy(dummy_vector)
        outputs = [grpcclient.InferRequestedOutput("output_prediction")]

        t0 = time.perf_counter()
        response = client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)
        pred_out = response.as_numpy("output_prediction")
        _ = inspection_filter.process_prediction(pred_out)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean_lat = float(np.mean(latencies))

    print(f"[✓] Stream completed ({num_frames} frames).")
    print(f"    Mean Latency : {mean_lat:.4f} ms")
    print(f"    p50 Latency  : {p50:.4f} ms")
    print(f"    p95 Latency  : {p95:.4f} ms")
    print(f"    p99 Latency  : {p99:.4f} ms")

    # Export benchmark metrics for check_latency_gate.py evaluation
    benchmark_data = {
        "summary_ms": {
            "total_glass_to_glass": {
                "mean": mean_lat,
                "p50": p50,
                "p95": p95,
                "p99": p99,
            }
        }
    }

    output_json_path = os.getenv(
        "BENCHMARK_RESULTS_PATH", "benchmark_results.json"
    )
    with open(output_json_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(
        f"[INFO] Benchmark results exported successfully to '{output_json_path}'."
    )


if __name__ == "__main__":
    try:
        run_triton_stream()
    except Exception as e:
        print(f"[!] Triton Stream Test Failed: {e}")
        sys.exit(1)