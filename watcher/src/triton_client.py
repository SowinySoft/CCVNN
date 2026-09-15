import asyncio
import logging
import queue
import threading
from typing import Callable, Optional
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

logger = logging.getLogger("TritonDecoupledClient")


class AsyncTritonStreamClient:
    """Decoupled, non-blocking gRPC streaming client for Triton Inference Server.

    Supports asynchronous inference requests with dedicated callback handlers to
    prevent pipeline stalling during high-throughput multi-camera processing.
    """

    def __init__(self, url: str = "localhost:8001", verbose: bool = False):
        self.url = url
        self.verbose = verbose
        self.client: Optional[grpcclient.InferenceServerClient] = None
        self._is_streaming = False
        self._response_queue = queue.Queue()

    def connect(self):
        """Initializes the gRPC client connection."""
        try:
            self.client = grpcclient.InferenceServerClient(
                url=self.url, verbose=self.verbose
            )
            if not self.client.is_server_live():
                raise ConnectionError(
                    f"Triton server at {self.url} is not live."
                )
            logger.info(f"Connected to Triton gRPC server at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise

    def _stream_callback(self, result, error):
        """Non-blocking internal callback invoked by Triton gRPC thread upon output completion."""
        if error:
            logger.error(f"Triton stream inference error: {error}")
            self._response_queue.put(
                {"status": "ERROR", "error": str(error), "result": None}
            )
        else:
            self._response_queue.put(
                {"status": "SUCCESS", "error": None, "result": result}
            )

    def start_decoupled_stream(self, callback_handler: Optional[Callable] = None):
        """Starts the background gRPC stream connection."""
        if not self.client:
            self.connect()

        user_callback = callback_handler or self._stream_callback
        try:
            self.client.start_stream(callback=user_callback)
            self._is_streaming = True
            logger.info("Decoupled Triton gRPC stream initiated successfully.")
        except InferenceServerException as e:
            logger.error(f"Failed to start Triton stream: {e}")
            raise

    def async_infer(
        self,
        model_name: str,
        input_data: np.ndarray,
        input_name: str = "input_0",
        output_name: str = "output_0",
        sequence_id: int = 0,
        request_id: str = "",
    ):
        """Submits an asynchronous inference request through the decoupled stream."""
        if not self._is_streaming:
            raise RuntimeError(
                "Stream is not active. Call start_decoupled_stream() first."
            )

        # Build Triton Input Tensor
        inputs = [grpcclient.InferInput(input_name, input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        # Build Triton Requested Output Tensor
        outputs = [grpcclient.InferRequestedOutput(output_name)]

        # Submit non-blocking async payload
        self.client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id,
            sequence_id=sequence_id,
        )

    def poll_result(self, block: bool = False, timeout: float = 0.05):
        """Polls processed inference results from the callback queue."""
        try:
            return self._response_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def stop_decoupled_stream(self):
        """Gracefully closes the active gRPC stream."""
        if self.client and self._is_streaming:
            self.client.stop_stream()
            self._is_streaming = False
            logger.info("Triton gRPC stream stopped cleanly.")


# Quick standalone validation
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = AsyncTritonStreamClient(url="localhost:8001")
    logger.info("AsyncTritonStreamClient module compiled successfully.")