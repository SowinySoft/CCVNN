import logging
import numpy as np
import tritonclient.grpc as grpcclient
from typing import List, Dict, Any, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TritonClient")

class TritonStreamingClient:
    def __init__(self, triton_url: str = "127.0.0.1:8001", model_name: str = "yolov8_safety"):
        self.triton_url = triton_url
        self.model_name = model_name
        self.client = grpcclient.InferenceServerClient(url=self.triton_url)

    def check_health(self) -> bool:
        """Verifies Triton gRPC endpoint accessibility and model readiness."""
        try:
            is_live = self.client.is_server_live()
            is_ready = self.client.is_model_ready(self.model_name)
            logger.info(f"Triton Live: {is_live} | Model [{self.model_name}] Ready: {is_ready}")
            return is_live and is_ready
        except Exception as e:
            logger.error(f"Triton connection failed: {e}")
            return False

    def infer_frame(self, frame_tensor: np.ndarray, class_mapping: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Sends preprocessed image tensor to Triton gRPC and parses detection results.
        Returns detection list format compatible with WatcherService.
        """
        inputs = [
            grpcclient.InferInput("images", frame_tensor.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(frame_tensor)

        outputs = [
            grpcclient.InferRequestedOutput("output0")
        ]

        response = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        output_data = response.as_numpy("output0")  # Shape: [1, N, 6] -> [x1, y1, x2, y2, conf, class_id]

        detections = []
        if output_data is not None and len(output_data) > 0:
            for det in output_data[0]:
                if len(det) >= 6:
                    x1, y1, x2, y2, conf, class_id = det[:6]
                    class_name = class_mapping.get(int(class_id), "unknown")
                    detections.append({
                        "class_name": class_name,
                        "confidence": float(conf),
                        "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
                        "tracking_id": f"track_{int(class_id)}"
                    })

        return detections