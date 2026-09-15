from watcher.src.engine import DynamicSeverityEngine

def test_severity_engine():
    engine = DynamicSeverityEngine("configs/severity_config.yaml")

    # Test Fire/Smoke (Requires 3 consecutive frames @ >= 0.70 confidence)
    assert engine.process_detection("fire_smoke", 0.75, "cam1_obj1") is None  # Frame 1
    assert engine.process_detection("fire_smoke", 0.80, "cam1_obj1") is None  # Frame 2
    
    rule = engine.process_detection("fire_smoke", 0.82, "cam1_obj1")          # Frame 3 (Trigger)
    assert rule is not None
    assert rule.severity == "CRITICAL"
    print("✓ Fire/Smoke debouncing test passed successfully.")

    # Test Low-Confidence Drop Reset
    engine.reset_tracker("cam1_obj1", "fire_smoke")
    assert engine.process_detection("fire_smoke", 0.75, "cam1_obj1") is None  # Frame 1
    assert engine.process_detection("fire_smoke", 0.40, "cam1_obj1") is None  # Drop below threshold (Reset)
    assert engine.process_detection("fire_smoke", 0.75, "cam1_obj1") is None  # Frame 1 (Restarted)
    print("✓ Debouncing reset on confidence drop test passed successfully.")

if __name__ == "__main__":
    test_severity_engine()