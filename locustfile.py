from locust import HttpUser, task, between
import json

class CCVNNUser(HttpUser):
    # Aggressive wait time to match high-concurrency profiling requirements
    wait_time = between(0.001, 0.01)

    @task(3)
    def health_check(self):
        """Verifies server readiness and response latency."""
        self.client.get("/v2/health/ready", name="/v2/health/ready")

    @task(6)
    def predict_single(self):
        """Tests single-vector inference performance."""
        payload = {
            "inputs": [
                {
                    "name": "v_input_17",
                    "shape": [1, 17],
                    "datatype": "FP32",
                    "data": [0.5] * 17
                }
            ]
        }
        self.client.post(
            "/v2/models/ccvnn_v17_backbone/infer",
            json=payload,
            name="/v2/infer-single"
        )

    @task(3)
    def predict_batch(self):
        """Tests batched inference performance under concurrency."""
        # Batch size of 4 matching your dynamic batching config
        batch_size = 4
        payload = {
            "inputs": [
                {
                    "name": "v_input_17",
                    "shape": [batch_size, 17],
                    "datatype": "FP32",
                    "data": [0.5] * (17 * batch_size)
                }
            ]
        }
        self.client.post(
            "/v2/models/ccvnn_v17_backbone/infer",
            json=payload,
            name="/v2/infer-batch"
        )