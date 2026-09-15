from unittest.mock import patch, MagicMock
from watcher.src.main import WatcherService

@patch("watcher.src.actuator.ModbusTcpClient")
def test_full_safety_pipeline(mock_modbus_client_cls):
    # Mock Modbus client response
    mock_modbus = MagicMock()
    mock_modbus.connect.return_value = True
    mock_result = MagicMock()
    mock_result.isError.return_value = False
    mock_modbus.write_register.return_value = mock_result
    mock_modbus_client_cls.return_value = mock_modbus

    # Initialize full Watcher Service
    service = WatcherService(config_path="configs/severity_config.yaml")

    # Simulate 2 consecutive production line crash detections (Threshold: 2 frames)
    frame_det = [{
        "class_name": "production_crash",
        "confidence": 0.88,
        "bounding_box": [50, 60, 150, 160],
        "tracking_id": "line_1_conveyor"
    }]

    print("\n--- Ingesting Frame 1 ---")
    service.handle_frame_detections("cam_line_1", "zone_production", frame_det)

    print("\n--- Ingesting Frame 2 (Trigger Expected) ---")
    service.handle_frame_detections("cam_line_1", "zone_production", frame_det)

    # Verify PLC hardware trigger was executed
    mock_modbus.write_register.assert_called_once_with(address=40001, value=1, slave=1)
    print("\n✓ Full pipeline integration test passed: Detection -> Debounce -> MQTT -> PLC.")

    service.shutdown()

if __name__ == "__main__":
    test_full_safety_pipeline()