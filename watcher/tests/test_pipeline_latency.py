from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List
import numpy as np

# Dynamic path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / "watcher" / "src"
CONFIG_PATH = PROJECT_ROOT / "config" / "watcher_stress_test_rules.yaml"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from watcher.src.telemetry import CameraHealthMonitor
from WatcherEngine import AlertAction, WatcherEngine

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PipelineLatencyBenchmark")


@dataclass
class StageLatencyRecord:
    ingest_ms: float
    inference_ms: float
    evaluation_ms: float
    actuation_ms: float
    total_e2e_ms: float


class MockTritonClient:
    """Low-overhead mock client simulating Triton gRPC stream latency overhead (1.5 - 4.0ms) if server is offline."""

    def mock_infer(self, frame: np.ndarray) -> bool:
        start = time.perf_counter_ns()
        # Simulate neural network tensor preprocessing & mock inference execution delay
        _ = np.mean(frame)
        time.sleep(0.0025)  # 2.5 ms synthetic model execution latency
        return True


class PipelineLatencyBenchmark:

    def __init__(
        self,
        iterations: int = 100,
        cam_id: str = "cam_01",
        sla_target_ms: float = 50.0,
    ):
        self.iterations = iterations
        self.cam_id = cam_id
        self.sla_target_ms = sla_target_ms
        self.records: List[StageLatencyRecord] = []

        self.health_monitor = CameraHealthMonitor()
        self.engine = WatcherEngine(config_path=str(CONFIG_PATH))
        self.triton = MockTritonClient()

    def generate_test_frame(self) -> np.ndarray:
        y, x = np.ogrid[:480, :640]
        mask = ((x // 16) + (y // 16)) % 2 == 0
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        frame[mask] = 40
        return frame

    def run_benchmark(self):
        print("\n=======================================================")
        print(" RUNNING END-TO-END PIPELINE LATENCY BENCHMARK")
        print(
            f" Iterations: {self.iterations} | Target SLA: < {self.sla_target_ms} ms"
        )
        print("=======================================================\n")

        # Warmup execution pass
        dummy_frame = self.generate_test_frame()
        self.health_monitor.analyze_frame(dummy_frame)
        self.engine.evaluate_rule(self.cam_id, True)

        for i in range(self.iterations):
            # Stage 1: Ingestion & Telemetry
            t0 = time.perf_counter_ns()
            frame = self.generate_test_frame()
            _ = self.health_monitor.analyze_frame(frame)
            t1 = time.perf_counter_ns()

            # Stage 2: Inference Stream
            hazard_detected = self.triton.mock_infer(frame)
            t2 = time.perf_counter_ns()

            # Stage 3: Watcher State Evaluation
            action = self.engine.evaluate_rule(self.cam_id, hazard_detected)
            t3 = time.perf_counter_ns()

            # Stage 4: Actuation / Dispatch Logic
            if action != AlertAction.NONE:
                # Simulate Modbus TCP register write delay
                time.sleep(0.001)  # 1.0 ms network roundtrip
            t4 = time.perf_counter_ns()

            ingest_ms = (t1 - t0) / 1e6
            inference_ms = (t2 - t1) / 1e6
            evaluation_ms = (t3 - t2) / 1e6
            actuation_ms = (t4 - t3) / 1e6
            total_e2e_ms = (t4 - t0) / 1e6

            self.records.append(
                StageLatencyRecord(
                    ingest_ms=ingest_ms,
                    inference_ms=inference_ms,
                    evaluation_ms=evaluation_ms,
                    actuation_ms=actuation_ms,
                    total_e2e_ms=total_e2e_ms,
                )
            )

        self.print_results()

    def print_results(self):
        e2e_times = [r.total_e2e_ms for r in self.records]
        ingest_times = [r.ingest_ms for r in self.records]
        infer_times = [r.inference_ms for r in self.records]
        eval_times = [r.evaluation_ms for r in self.records]
        actuate_times = [r.actuation_ms for r in self.records]

        p50 = np.percentile(e2e_times, 50)
        p95 = np.percentile(e2e_times, 95)
        p99 = np.percentile(e2e_times, 99)
        max_time = np.max(e2e_times)

        print("=======================================================")
        print(" LATENCY BREAKDOWN BY PIPELINE STAGE (Mean ms)")
        print("=======================================================")
        print(f" 1. Ingestion & Telemetry:  {np.mean(ingest_times):.3f} ms")
        print(f" 2. Triton Model Inference: {np.mean(infer_times):.3f} ms")
        print(f" 3. Watcher Rule Eval:      {np.mean(eval_times):.3f} ms")
        print(f" 4. PLC/MQTT Actuation:     {np.mean(actuate_times):.3f} ms")
        print("-------------------------------------------------------")
        print(" END-TO-END LATENCY PERCENTILES")
        print("-------------------------------------------------------")
        print(f" P50 (Median):               {p50:.3f} ms")
        print(f" P95:                        {p95:.3f} ms")
        print(f" P99:                        {p99:.3f} ms")
        print(f" Maximum Latency:            {max_time:.3f} ms")
        print("=======================================================")

        if p99 < self.sla_target_ms:
            print(
                f"✓ SUCCESS: P99 End-to-End Latency ({p99:.2f} ms) is within < {self.sla_target_ms} ms SLA."
            )
        else:
            print(
                f"✗ FAILURE: P99 Latency ({p99:.2f} ms) exceeded target SLA of {self.sla_target_ms} ms!"
            )


if __name__ == "__main__":
    bench = PipelineLatencyBenchmark(iterations=100, sla_target_ms=50.0)
    bench.run_benchmark()