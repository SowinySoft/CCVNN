import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from typing import List, Tuple
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TritonBatchingBenchmark")

try:
    import tritonclient.grpc as grpcclient
    TRITON_CLIENT_AVAILABLE = True
except ImportError:
    TRITON_CLIENT_AVAILABLE = False


def send_inference_request(
    client: "grpcclient.InferenceServerClient",
    model_name: str,
    dummy_input: np.ndarray,
    request_id: str,
) -> Tuple[str, float]:
    """Sends a single inference request to Triton server via gRPC and returns latency in ms."""
    inputs = [grpcclient.InferInput("input__0", dummy_input.shape, "FP32")]
    inputs[0].set_data_from_numpy(dummy_input)
    outputs = [grpcclient.InferRequestedOutput("output__0")]

    start_time = time.perf_counter()
    client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        request_id=request_id,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return request_id, latency_ms


def simulate_dynamic_batching(num_concurrent_cams: int, requests_per_cam: int) -> None:
    """Simulates dynamic batching queue delay (max_batch_size: 8, max_queue_delay: 5000us)."""
    total_requests = num_concurrent_cams * requests_per_cam
    batch_size = 8
    queue_delay_sec = 0.005  # 5000us
    compute_time_per_batch = 0.008  # 8ms GPU compute time

    batches_needed = (total_requests + batch_size - 1) // batch_size
    simulated_total_time = batches_needed * (queue_delay_sec + compute_time_per_batch)
    simulated_fps = total_requests / simulated_total_time
    avg_lat = (queue_delay_sec + compute_time_per_batch) * 1000.0

    logger.info("--- Simulated Dynamic Batching Profile ---")
    logger.info("Batch Config      : max_batch_size=8, max_queue_delay=5000us")
    logger.info(f"Total Requests    : {total_requests}")
    logger.info(f"Simulated FPS     : {simulated_fps:.2f} frames/sec")
    logger.info(f"Simulated Latency : {avg_lat:.2f} ms per batch")


def run_benchmark(
    server_url: str,
    model_name: str,
    num_concurrent_cams: int,
    requests_per_cam: int,
    force_sim: bool = False,
) -> None:
    """Executes multi-threaded concurrency benchmark against Triton gRPC or simulated backend."""
    logger.info("--- Starting Triton Dynamic Batching Benchmark ---")
    logger.info(f"Server Target: {server_url} | Model: {model_name}")
    logger.info(
        f"Concurrency  : {num_concurrent_cams} workers | Requests/Worker: {requests_per_cam}"
    )

    if force_sim or not TRITON_CLIENT_AVAILABLE:
        logger.warning("Running simulated batching benchmark.")
        simulate_dynamic_batching(num_concurrent_cams, requests_per_cam)
        return

    try:
        client = grpcclient.InferenceServerClient(url=server_url)
        if not client.is_server_ready():
            logger.warning(
                f"Triton server at {server_url} is not ready. Falling back to simulation mode."
            )
            simulate_dynamic_batching(num_concurrent_cams, requests_per_cam)
            return

        if not client.is_model_ready(model_name):
            logger.warning(
                f"Model '{model_name}' is not loaded on Triton server. Falling back to simulation mode."
            )
            simulate_dynamic_batching(num_concurrent_cams, requests_per_cam)
            return

    except Exception as e:
        logger.warning(
            f"Failed to connect to Triton gRPC endpoint ({server_url}): {e}. Falling back to simulation mode."
        )
        simulate_dynamic_batching(num_concurrent_cams, requests_per_cam)
        return

    dummy_frame = np.random.randn(1, 3, 224, 224).astype(np.float32)
    total_requests = num_concurrent_cams * requests_per_cam

    latencies: List[float] = []
    start_global = time.perf_counter()

    def worker_loop(cam_idx: int) -> List[float]:
        worker_latencies = []
        for req_idx in range(requests_per_cam):
            req_id = f"cam_{cam_idx:03d}_req_{req_idx}"
            _, lat = send_inference_request(client, model_name, dummy_frame, req_id)
            worker_latencies.append(lat)
        return worker_latencies

    with ThreadPoolExecutor(max_workers=num_concurrent_cams) as executor:
        futures = [
            executor.submit(worker_loop, cam_i) for cam_i in range(num_concurrent_cams)
        ]
        for f in as_completed(futures):
            latencies.extend(f.result())

    total_time = time.perf_counter() - start_global
    fps = total_requests / total_time
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))

    logger.info("--- Triton Dynamic Batching Benchmark Results ---")
    logger.info(f"Total Requests Processed : {total_requests}")
    logger.info(f"Total Time Elapsed       : {total_time:.4f} s")
    logger.info(f"Throughput (FPS)         : {fps:.2f} frames/sec")
    logger.info(f"Mean Latency             : {avg_latency:.2f} ms")
    logger.info(f"P95 Latency              : {p95_latency:.2f} ms")
    logger.info(f"P99 Latency              : {p99_latency:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Dynamic Batching Benchmark")
    parser.add_argument("--url", default="localhost:8001", help="Triton gRPC server URL")
    parser.add_argument("--model", default="snn_classifier", help="Model name")
    parser.add_argument(
        "--cams", type=int, default=32, help="Number of concurrent camera workers"
    )
    parser.add_argument(
        "--requests", type=int, default=100, help="Requests per camera worker"
    )
    parser.add_argument("--sim", action="store_true", help="Force simulation mode")
    args = parser.parse_args()

    run_benchmark(args.url, args.model, args.cams, args.requests, force_sim=args.sim)