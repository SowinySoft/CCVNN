import pytest
import numpy as np
from unittest.mock import MagicMock
from src.ccvnn.rule_engine.multifactor_engine import MultiFactorRuleEngine
from src.ccvnn.pipeline.actuation_bridge import ActuationBridge


def test_actuation_bridge_mqtt_dispatch():
    rule_engine = MultiFactorRuleEngine(hazard_threshold=0.75, debounce_consecutive_frames=2)
    bridge = ActuationBridge(rule_engine=rule_engine)
    
    # Mock MQTT client
    bridge.client = MagicMock()

    hazard_payload = {
        "num_detections": np.array([[1]], dtype=np.int32),
        "detection_scores": np.zeros((1, 100), dtype=np.float32),
        "detection_classes": np.zeros((1, 100), dtype=np.int32),
        "output_hazard": np.array([[0.88]], dtype=np.float32),
    }

    # Frame 1: Pending debounce -> MQTT publish NOT called
    res1 = bridge.process_and_publish(hazard_payload)
    assert res1["trigger_actuation"] is False
    bridge.client.publish.assert_not_called()

    # Frame 2: Debounce threshold reached -> MQTT publish called
    res2 = bridge.process_and_publish(hazard_payload)
    assert res2["trigger_actuation"] is True
    bridge.client.publish.assert_called_once()