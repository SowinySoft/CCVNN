import time
import numpy as np
import torch
from collections import deque
import tritonclient.grpc as grpcclient

class IndustrialInspectionFilter:
    def __init__(self, window_size=5, margin_threshold=0.25):
        self.window = deque(maxlen=window_size)
        self.margin = margin_threshold
        self.last_stable_state = None

    def process_logits(self, logits_np: np.ndarray) -> int:
        probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()[0]
        margin = abs(probs[1] - probs[0])
        raw_pred = int(np.argmax(probs))

        self.window.append(raw_pred)
        majority_pred = 1 if sum(self.window) > (len(self.window) / 2) else 0

        if margin < self.margin and self.last_stable_state is not None:
            return self.last_stable_state

        self.last_stable_state = majority_pred
        return majority_pred

def run_triton_stream(server_url="localhost:8001", num_frames=1000):
    client = grpcclient.InferenceServerClient(url=server_url)
    inspection_filter = IndustrialInspectionFilter()

    latencies = []
    for i in range(num_frames):
        dummy_vector = np.random.rand(1, 9).astype(np.float32)
        inputs = [grpcclient.InferInput("x", [1, 9], "FP32")]
        inputs[0].set_data_from_numpy(dummy_vector)
        outputs = [grpcclient.InferRequestedOutput("output")]

        t0 = time.perf_counter()
        response = client.infer(model_name="ccvnn_inspector", inputs=inputs, outputs=outputs)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)
        logits = response.as_numpy("output")
        decision = inspection_filter.process_logits(logits)

    print(f"p50 Latency: {np.percentile(latencies, 50):.4f} ms")

if __name__ == "__main__":
    try:
        run_triton_stream()
    except Exception as e:
        print(f"[!] Triton Server Connection Check: {e}")
