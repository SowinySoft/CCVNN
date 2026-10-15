"""
Cyclotron Edge Frame Aggregator.
Synchronizes spatial entity streams and industrial Modbus telemetry.
"""

import time
from typing import Dict, Any, List, Optional
from watcher.schemas.vector_schema import VectorPayload, VectorSourceType, SpatialEntity


class EdgeFrameAggregator:
    def __init__(self, time_window_ms: float = 100.0):
        self.time_window_sec = time_window_ms / 1000.0
        self._latest_telemetry: Dict[str, Any] = {}

    def update_telemetry(self, plc_register_val: int, hazard_score: float) -> None:
        """Caches the latest PLC state for spatial alignment."""
        self._latest_telemetry = {
            "plc_register_value": plc_register_val,
            "hazard_score": hazard_score,
            "updated_at": time.time(),
        }

    def create_camera_vector(
        self,
        camera_id: str,
        entities: List[SpatialEntity],
        frame_dim: tuple = (1920, 1080),
    ) -> VectorPayload:
        """
        Constructs a unified VectorPayload merging spatial camera entities
        with real-time PLC state if available within the validity window.
        """
        now = time.time()
        telemetry_valid = (
            self._latest_telemetry
            and (now - self._latest_telemetry.get("updated_at", 0)) <= self.time_window_sec
        )

        return VectorPayload(
            vector_id=f"vec_{camera_id}_{int(now * 1000)}",
            timestamp=now,
            source_type=VectorSourceType.CAMERA_STREAM,
            source_id=camera_id,
            hazard_score=self._latest_telemetry.get("hazard_score", 0.0) if telemetry_valid else 0.0,
            plc_register_value=self._latest_telemetry.get("plc_register_value") if telemetry_valid else None,
            frame_width=frame_dim[0],
            frame_height=frame_dim[1],
            entities=entities,
        )