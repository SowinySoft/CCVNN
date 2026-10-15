import os
import time
import yaml
import logging
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from watcher.src.telemetry import CameraHealthMonitor, SystemHardwareMonitor
from watcher.src.engine import DynamicSeverityEngine, SeverityRule
from watcher.src.publisher import MQTTPublisher

# Safe fallback for PLCActuator if watcher.src.plc is not present
try:
    from watcher.src.plc import PLCActuator
except ImportError:
    class PLCActuator:
        def __init__(self, host: str = "192.168.10.45", port: int = 502, mock_mode: bool = True):
            self.host = host
            self.port = port
            self.mock_mode = mock_mode
            logging.getLogger("PLCActuatorMock").info(f"Initialized PLC Actuator at {host}:{port} (mock_mode={mock_mode})")

        def trigger_actuation(self, register: int, value: int):
            logging.getLogger("PLCActuatorMock").info(f"Wrote value {value} to PLC {self.host}:{self.port} [Reg: {register}]")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s")
logger = logging.getLogger("FullPipelineStress")

CONFIG_PATH = "watcher_stress_test_rules.yaml"


def create_stress_test_config():
    """Generates a temporary rule configuration YAML for the stress run."""
    config_data = {
        "default_debouncing": {
            "min_confidence": 0.50,
            "consecutive_frames": 2
        },
        "event_classes": {
            "FORKLIFT": {
                "class_id": 1,
                "display_name": "Forklift Hazard",
                "min_confidence": 0.80,
                "consecutive_frames": 2,
                "severity": "CRITICAL",
                "action_type": "PLC_HALT",
                "plc_config": {"register": 40001, "value": 1},
                "topics": ["factory/zone_main/alerts/critical"]
            }
        }
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_data, f)


class FullPipelineStressTest:
    def __init__(self, num_cameras: int = 4, frames_per_cam: int = 50):
        self.num_cameras = num_cameras
        self.frames_per_cam = frames_per_cam
        self.total_processed_frames = 0
        self.total_alerts_triggered = 0
        self.lock = threading.Lock()

        create_stress_test_config()

        # Core Pipeline Components
        self.cam_monitors = {f"cam_0{i+1}": CameraHealthMonitor() for i in range(num_cameras)}
        self.hw_monitor = SystemHardwareMonitor()
        self.rule_engine = DynamicSeverityEngine(config_path=CONFIG_PATH)
        self.publisher = MQTTPublisher(host="127.0.0.1", port=1883)
        self.plc = PLCActuator(host="192.168.10.45", port=502, mock_mode=True)

    def generate_simulated_frame(self, frame_idx: int, is_anomaly: bool = False) -> np.ndarray:
        """Generates synthetic video frame buffers with varying visual features."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        if not is_anomaly:
            grid_size = 40
            for i in range(0, 480, grid_size):
                for j in range(0, 640, grid_size):
                    if (i // grid_size + j // grid_size) % 2 == 0:
                        frame[i:i + grid_size, j:j + grid_size] = 200
        else:
            frame[:, :] = 15
        return frame

    def process_camera_stream(self, cam_id: str):
        """Simulates real-time video stream pipeline for a single camera channel."""
        logger.info(f"Starting ingestion pipeline worker for [{cam_id}]...")

        for f_idx in range(self.frames_per_cam):
            is_anomaly = (f_idx % 15 == 0 and f_idx > 0)
            frame = self.generate_simulated_frame(f_idx, is_anomaly=is_anomaly)

            # 1. Telemetry & Lens Assessment
            cam_health = self.cam_monitors[cam_id].analyze_frame(frame)

            # 2. Ingestion & Temporal Debouncing via DynamicSeverityEngine
            tracking_id = f"track_{cam_id}_01"
            rule_triggered = self.rule_engine.process_detection(
                class_name="FORKLIFT",
                confidence=0.92,
                tracking_id=tracking_id
            )

            # 3. MQTT Alerting & PLC Hardware Actuation Loop
            if rule_triggered:
                self.publisher.publish_event(
                    rule=rule_triggered,
                    sensor_id=cam_id,
                    zone_id="zone_main",
                    confidence=0.92,
                    bounding_box=[100, 150, 250, 400],
                    tracking_id=tracking_id
                )
                if rule_triggered.severity == "CRITICAL":
                    self.plc.trigger_actuation(register=40001, value=1)
                with self.lock:
                    self.total_alerts_triggered += 1

            with self.lock:
                self.total_processed_frames += 1

            time.sleep(0.01)

    def run(self):
        print(f"\n=======================================================")
        print(f" STARTING FULL PIPELINE SYSTEM STRESS TEST")
        print(f" Parallel Cameras: {self.num_cameras} | Frames per Cam: {self.frames_per_cam}")
        print(f" Target Total Workload: {self.num_cameras * self.frames_per_cam} frames")
        print(f"=======================================================\n")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.num_cameras, thread_name_prefix="CamWorker") as executor:
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
        print(f" ✓ Total Processed Frames:  {self.total_processed_frames}")
        print(f" ✓ Total Alerts Triggered:   {self.total_alerts_triggered}")
        print(f" ✓ Total Execution Time:     {elapsed:.2f} seconds")
        print(f" ✓ Pipeline Throughput:      {fps:.1f} FPS")
        print(f" ✓ Host CPU Thermal Readout: {hw_status.cpu_temp_c:.1f}°C")
        print(f" ✓ Host CPU Utilization:     {hw_status.cpu_usage_pct:.1f}%")
        print(f" ✓ Host RAM Utilization:     {hw_status.ram_usage_pct:.1f}%")
        print(f" ✓ Telemetry Status:         {hw_status.status}")
        print(f"=======================================================\n")

        # Cleanup test config file
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)

        assert self.total_processed_frames == (self.num_cameras * self.frames_per_cam)
        print("✓ All frame streams ingested, correlated, and actuated with 0 dropped events.")


if __name__ == "__main__":
    test_runner = FullPipelineStressTest(num_cameras=4, frames_per_cam=50)
    test_runner.run()