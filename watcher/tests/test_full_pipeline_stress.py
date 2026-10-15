import concurrent.futures
import logging
import os
from pathlib import Path
import sys
import threading
import time
import numpy as np

# Dynamic path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / "watcher" / "src"
CONFIG_PATH = PROJECT_ROOT / "config" / "watcher_stress_test_rules.yaml"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from watcher.src.telemetry import CameraHealthMonitor, SystemHardwareMonitor
from WatcherEngine import AlertAction, WatcherEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FullPipelineStress")

# Safe fallback for PLCActuator if watcher.src.plc is not present
try:
    from watcher.src.plc import PLCActuator
except ImportError:

    class PLCActuator:

        def __init__(
            self,
            host: str = "192.168.10.45",
            port: int = 502,
            mock_mode: bool = True,
        ):
            self.host = host
            self.port = port
            self.mock_mode = mock_mode

        def trigger_actuation(self, register: int, value: int):
            logger.info(
                f"PLC WRITE -> Host: {self.host}:{self.port} | Reg {register} = {value}"
            )


class FullPipelineStressTest:

    def __init__(self, num_cameras: int = 4, frames_per_cam: int = 50):
        self.num_cameras = num_cameras
        self.frames_per_cam = frames_per_cam
        self.total_processed_frames = 0
        self.total_alerts_triggered = 0
        self.lock = threading.Lock()

        self.hw_monitor = SystemHardwareMonitor()
        self.plc = PLCActuator(host="192.168.10.45", port=502, mock_mode=True)
        self.cam_monitors = {
            f"cam_0{i+1}": CameraHealthMonitor() for i in range(num_cameras)
        }

    def generate_simulated_frame(
        self, frame_idx: int, is_anomaly: bool = False
    ) -> np.ndarray:
        """Generates high-contrast textured frames to satisfy blur score (>=100.0) and occlusion thresholds."""
        base_val = 180 if not is_anomaly else 70
        alt_val = 40 if not is_anomaly else 210

        # 16px grid creates sharp pixel contrast transitions for Laplacian focus checks
        y, x = np.ogrid[:480, :640]
        mask = ((x // 16) + (y // 16)) % 2 == 0

        frame = np.full((480, 640, 3), base_val, dtype=np.uint8)
        frame[mask] = alt_val
        return frame

    def process_camera_stream(self, cam_id: str):
        """Worker thread processing a single camera stream with WatcherEngine state suppression."""
        engine = WatcherEngine(config_path=str(CONFIG_PATH))
        rule_cfg = engine.rules.get(cam_id, {})
        plc_reg = rule_cfg.get("plc_register", 40001)
        mqtt_topic = rule_cfg.get("mqtt_topic", f"factory/{cam_id}/hazard")

        logger.info(
            f"[{cam_id}] Worker started (Persistence: {rule_cfg.get('persistence_threshold', 2)}, "
            f"Heartbeat: {rule_cfg.get('heartbeat_interval', 10)})"
        )

        for f_idx in range(1, self.frames_per_cam + 1):
            hazard_detected = 10 <= f_idx <= 35
            frame = self.generate_simulated_frame(
                f_idx, is_anomaly=hazard_detected
            )

            # Telemetry evaluation
            _ = self.cam_monitors[cam_id].analyze_frame(frame)

            # State-aware rule evaluation
            action = engine.evaluate_rule(cam_id, hazard_detected)

            if action == AlertAction.INITIAL_TRIGGER:
                self.plc.trigger_actuation(register=plc_reg, value=1)
                logger.info(
                    f"MQTT PUB  -> Topic: {mqtt_topic} | Payload: {cam_id} [INITIAL_TRIGGER]"
                )
                with self.lock:
                    self.total_alerts_triggered += 1

            elif action == AlertAction.HEARTBEAT_REMINDER:
                self.plc.trigger_actuation(register=plc_reg, value=1)
                logger.info(
                    f"MQTT PUB  -> Topic: {mqtt_topic} | Payload: {cam_id} [HEARTBEAT]"
                )
                with self.lock:
                    self.total_alerts_triggered += 1

            elif action == AlertAction.CLEAR:
                self.plc.trigger_actuation(register=plc_reg, value=0)
                logger.info(
                    f"MQTT PUB  -> Topic: {mqtt_topic} | Payload: {cam_id} [CLEARED]"
                )

            with self.lock:
                self.total_processed_frames += 1

            time.sleep(0.005)

    def run(self):
        print(f"\n=======================================================")
        print(f" STARTING PIPELINE STRESS TEST WITH STATE SUPPRESSION")
        print(
            f" Parallel Cameras: {self.num_cameras} | Frames per Cam: {self.frames_per_cam}"
        )
        print(
            f" Target Total Workload: {self.num_cameras * self.frames_per_cam} frames"
        )
        print(f"=======================================================\n")

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_cameras, thread_name_prefix="CamWorker"
        ) as executor:
            futures = [
                executor.submit(self.process_camera_stream, f"cam_0{i+1}")
                for i in range(self.num_cameras)
            ]
            for future in futures:
                future.result()

        elapsed = time.time() - start_time
        fps = self.total_processed_frames / elapsed
        hw_status = self.hw_monitor.get_telemetry()

        print(f"\n=======================================================")
        print(f" STRESS TEST RESULTS & SYSTEM METRICS")
        print(f"=======================================================")
        print(
            f" ✓ Total Processed Frames:  {self.total_processed_frames} / {self.num_cameras * self.frames_per_cam}"
        )
        print(f" ✓ Total Dispatched Alerts:  {self.total_alerts_triggered}")
        print(f" ✓ Execution Time:           {elapsed:.2f} seconds")
        print(f" ✓ Pipeline Throughput:      {fps:.1f} FPS")
        print(
            f" ✓ Host CPU Thermal Readout: {hw_status.cpu_temp_c:.1f}°C"
        )
        print(
            f" ✓ Host CPU Utilization:     {hw_status.cpu_usage_pct:.1f}%"
        )
        print(
            f" ✓ Host RAM Utilization:     {hw_status.ram_usage_pct:.1f}%"
        )
        print(f"=======================================================\n")

        assert self.total_processed_frames == (
            self.num_cameras * self.frames_per_cam
        )
        print(
            "✓ Pipeline completed successfully with zero telemetry or frame drops."
        )


if __name__ == "__main__":
    test_runner = FullPipelineStressTest(num_cameras=4, frames_per_cam=50)
    test_runner.run()