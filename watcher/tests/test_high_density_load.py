from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from watcher.src.WatcherEngine import AlertAction, WatcherEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HighDensityStressTest")


def simulate_camera_lifecycle(
    engine: WatcherEngine, cam_id: str, frame_count: int
):
    """Simulates a camera thread sending raw anomaly frames into the Watcher Engine."""
    # 1. Dynamic registration
    engine.register_camera(cam_id)

    actions_recorded = []
    start_time = time.perf_counter()

    # 2. Simulate anomaly stream (alternating normal -> anomaly -> persistent -> clear)
    for frame_idx in range(frame_count):
        raw_anomaly = True if 10 <= frame_idx <= 25 else False
        action = engine.evaluate_rule(cam_id, raw_anomaly)
        if action != AlertAction.NONE:
            actions_recorded.append((frame_idx, action))

    elapsed = time.perf_counter() - start_time
    return cam_id, elapsed, len(actions_recorded)


def run_stress_test(num_cameras: int = 100, frames_per_camera: int = 200):
    logger.info(
        f"--- Starting High-Density Stress Test ({num_cameras} streams x {frames_per_camera} frames) ---"
    )

    config_path = "config/watcher_stress_test_rules.yaml"
    engine = WatcherEngine(config_path=config_path)

    total_frames = num_cameras * frames_per_camera
    start_global = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(
                simulate_camera_lifecycle,
                engine,
                f"cam_stress_{i:03d}",
                frames_per_camera,
            )
            for i in range(num_cameras)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    total_duration = time.perf_counter() - start_global
    throughput = total_frames / total_duration
    avg_lat_ms = (total_duration / total_frames) * 1000

    logger.info("--- Stress Test Results ---")
    logger.info(f"Registered Cameras : {len(engine.registered_cameras)}")
    logger.info(f"Assigned PLC Regs  : {len(engine.assigned_registers)}")
    logger.info(f"Total Frames Processed: {total_frames}")
    logger.info(f"Total Execution Time  : {total_duration:.4f} s")
    logger.info(f"System Throughput     : {throughput:.2f} evals/sec")
    logger.info(f"Avg Latency per Frame : {avg_lat_ms:.4f} ms")

    # Verification Checks
    assert (
        len(engine.registered_cameras) == num_cameras + 1
    ), "Camera count mismatch (including static cam_01)!"
    assert len(engine.assigned_registers) == len(
        engine.registered_cameras
    ), "PLC Register collision detected under stress load!"

    logger.info(
        "\n✓ SUCCESS: High-density stress test completed with zero thread locks or register collisions."
    )


if __name__ == "__main__":
    run_stress_test(num_cameras=100, frames_per_camera=200)
    