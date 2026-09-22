"""
Cyclotron Multi-Stream Synthetic Client.
Simulates multi-camera frame ingestion alongside Modbus PLC telemetry streams.
"""

import asyncio
import logging
import time
from watcher.schemas.vector_schema import VectorPayload, VectorSourceType, SpatialEntity
from watcher.ingestion.buffer import MultiVectorIngestionBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiStreamClient")


async def generate_synthetic_stream(buffer: MultiVectorIngestionBuffer, num_frames: int = 50):
    logger.info(f"Starting synthetic multi-stream generation ({num_frames} frames)...")
    
    for i in range(num_frames):
        entities = [
            SpatialEntity(
                label="person",
                confidence=0.85 if i % 2 == 0 else 0.45,
                bbox=[0.1, 0.2, 0.5, 0.6],
                class_id=0
            )
        ]
        
        payload = VectorPayload(
            vector_id=f"synth_vec_{i}_{int(time.time() * 1000)}",
            timestamp=time.time(),
            source_type=VectorSourceType.CAMERA_STREAM,
            source_id="cam_01",
            hazard_score=0.82 if i % 3 == 0 else 0.15,
            plc_register_value=40001,
            frame_width=1920,
            frame_height=1080,
            entities=entities
        )
        
        await buffer.push(payload)
        await asyncio.sleep(0.01)  # 100 Hz ingestion pulse

    logger.info("Synthetic multi-stream ingestion completed successfully.")