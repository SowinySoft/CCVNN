import numpy as np
from unittest.mock import MagicMock, patch
from watcher.src.main import WatcherService


@patch("watcher.src.actuator.ModbusTcpClient")
@patch("watcher.src.main.DynamicSeverityEngine")
def test_full_safety_pipeline(mock_engine_cls, mock_modbus_client_cls):
    """Verifies end-to-end pipeline: frame ingestion -> tracking -> engine debouncing -> PLC actuation."""
    # 1. Setup Mock PLC Client
    mock_modbus = MagicMock()
    mock_modbus.connect.return_value = True
    mock_result = MagicMock()
    mock_result.isError.return_value = False
    mock_modbus.write_register.return_value = mock_result
    mock_modbus_client_cls.return_value = mock_modbus

    # 2. Setup Mock Severity Engine Rule Output
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    mock_rule = MagicMock()
    mock_rule.display_name = "Production Crash"
    mock_rule.severity = "CRITICAL"
    mock_rule.action_type = "PLC_RELAY_TRIP"
    mock_rule.rule = "Production Crash"
    mock_rule.rule_id = "CRASH_001"
    mock_rule.plc_config = {
        "ip": "192.168.10.45",
        "port": 502,
        "register": 40001,
        "value": 1,
    }

    # First frame returns None (debouncing counter = 1)
    # Second frame returns rule (threshold met -> trigger)
    mock_engine.process_detection.side_effect = [None, mock_rule]

    # 3. Initialize Watcher Service
    service = WatcherService(config_path="configs/severity_config.yaml")

    frame_det = [
        {
            "class_name": "production_crash",
            "confidence": 0.88,
            "bounding_box": [50, 60, 150, 160],
            "tracking_id": "line_1_conveyor",
        }
    ]

    # Frame 1: Ingest (Accumulate state)
    service.handle_frame_detections("cam_line_1", "zone_production", frame_det)
    mock_modbus.write_register.assert_not_called()

    # Frame 2: Ingest (Debounce threshold met -> Trigger)
    service.handle_frame_detections("cam_line_1", "zone_production", frame_det)

    if hasattr(service, "shutdown"):
        service.shutdown()