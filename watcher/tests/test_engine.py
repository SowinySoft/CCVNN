import pytest
from watcher.engine import DynamicSeverityEngine


def test_severity_engine():
    engine = DynamicSeverityEngine("configs/severity_config.yaml")

    # Feed consecutive detection frames until threshold rule triggers
    rule = None
    for conf in [0.75, 0.80, 0.82, 0.85, 0.90]:
        res = engine.process_detection("fire_smoke", conf, "cam1_obj1")
        if res is not None:
            rule = res
            break

    assert rule is not None, "Severity engine failed to trigger rule after consecutive frames."