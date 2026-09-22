import pytest
import numpy as np
from src.ccvnn.rule_engine.multifactor_engine import MultiFactorRuleEngine


def test_rule_engine_normal_state():
    engine = MultiFactorRuleEngine(hazard_threshold=0.75, min_vision_confidence=0.5)
    mock_output = {
        "num_detections": np.array([[0]], dtype=np.int32),
        "detection_scores": np.zeros((1, 100), dtype=np.float32),
        "detection_classes": np.zeros((1, 100), dtype=np.int32),
        "output_hazard": np.array([[0.1]], dtype=np.float32),
    }
    res = engine.evaluate(mock_output)
    assert res["raw_trigger"] is False
    assert res["trigger_actuation"] is False
    assert res["rule_matched"] == "NORMAL_OPERATIONAL"


def test_rule_engine_critical_hazard_immediate():
    # Set debounce_consecutive_frames=1 for immediate actuation on 1st frame
    engine = MultiFactorRuleEngine(
        hazard_threshold=0.75, 
        min_vision_confidence=0.5, 
        debounce_consecutive_frames=1
    )
    mock_output = {
        "num_detections": np.array([[1]], dtype=np.int32),
        "detection_scores": np.zeros((1, 100), dtype=np.float32),
        "detection_classes": np.zeros((1, 100), dtype=np.int32),
        "output_hazard": np.array([[0.85]], dtype=np.float32),
    }
    res = engine.evaluate(mock_output)
    assert res["raw_trigger"] is True
    assert res["trigger_actuation"] is True
    assert res["rule_matched"] == "DEBOUNCED_CRITICAL_HAZARD"


def test_debouncer_requires_consecutive_frames():
    engine = MultiFactorRuleEngine(hazard_threshold=0.75, debounce_consecutive_frames=3)
    
    hazard_payload = {
        "num_detections": np.array([[0]], dtype=np.int32),
        "detection_scores": np.zeros((1, 100), dtype=np.float32),
        "detection_classes": np.zeros((1, 100), dtype=np.int32),
        "output_hazard": np.array([[0.85]], dtype=np.float32),
    }

    # Frame 1: Raw trigger active, debounced actuation False
    res1 = engine.evaluate(hazard_payload)
    assert res1["raw_trigger"] is True
    assert res1["trigger_actuation"] is False
    assert res1["consecutive_frames"] == 1

    # Frame 2: Pending debounce
    res2 = engine.evaluate(hazard_payload)
    assert res2["trigger_actuation"] is False
    assert res2["consecutive_frames"] == 2

    # Frame 3: Debounce threshold met -> Actuation True
    res3 = engine.evaluate(hazard_payload)
    assert res3["trigger_actuation"] is True
    assert res3["consecutive_frames"] == 3
    assert res3["rule_matched"] == "DEBOUNCED_CRITICAL_HAZARD"
