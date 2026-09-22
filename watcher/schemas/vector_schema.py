"""
Cyclotron Multi-Factor Vector Vision Schema Definition.
Handles concurrent high-resolution image tensors, spatial bounding entities,
and Modbus PLC industrial telemetry.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import time


class VectorSourceType(str, Enum):
    CAMERA_STREAM = "camera_stream"
    TELEMETRY_MODBUS = "telemetry_modbus"
    MQTT_EVENT = "mqtt_event"
    SYNTHETIC_INJECTOR = "synthetic_injector"


class SpatialEntity(BaseModel):
    label: str
    confidence: float
    bbox: List[float] = Field(
        default_factory=list, 
        description="Bounding box [ymin, xmin, ymax, xmax] normalized [0, 1]"
    )
    class_id: int
    attributes: Dict[str, Any] = Field(default_factory=dict)


class VectorPayload(BaseModel):
    vector_id: str
    timestamp: float = Field(default_factory=time.time)
    source_type: VectorSourceType
    source_id: str
    
    # Industrial Telemetry Vector
    hazard_score: float = 0.0
    plc_register_value: Optional[int] = None
    
    # General Multi-Image Vision Vector
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    channels: Optional[int] = 3
    entities: List[SpatialEntity] = Field(default_factory=list)
    
    # Metadata & Custom Flags
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultiVectorBatch(BaseModel):
    batch_id: str
    created_at: float = Field(default_factory=time.time)
    vectors: List[VectorPayload]
    total_entities: int = 0

    def compute_aggregates(self) -> None:
        self.total_entities = sum(len(v.entities) for v in self.vectors)