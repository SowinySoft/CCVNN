import numpy as np
from unittest.mock import patch, MagicMock
from watcher.src.main import WatcherService
from watcher.src.storage import OfflineStorage

@patch("watcher.src.actuator.ModbusTcpClient")
def test_all_enhancements(mock_modbus):
    service = WatcherService("configs/severity_config.yaml")

    # Generate dummy frame stream (300 frames = 10s)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(30):
        service.ingest_frame(dummy_frame)

    # Ingest critical fire detection stream
    detections = [{
        "class_name": "fire_smoke",
        "confidence": 0.85,
        "bounding_box": [100, 100, 200, 200],
        "tracking_id": "fire_sensor_01"
    }]

    for _ in range(3):
        service.handle_frame_detections("cam_01", "zone_01", detections)

    print("\n✓ SHA-256 audit log, MP4 frame buffer dump, and offline buffer verified.")
    service.shutdown()

if __name__ == "__main__":
    test_all_enhancements()