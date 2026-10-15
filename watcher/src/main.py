import os
import signal
import sys
import time
import logging
import numpy as np
from typing import Dict, Any, List
from watcher.src.engine import DynamicSeverityEngine
from watcher.src.publisher import MQTTPublisher
from watcher.src.actuator import PLCActuator
from watcher.src.audit import AuditLogger, RollingFrameBuffer
from watcher.src.notifications import NotificationDispatcher
from watcher.src.tracker import SimpleObjectTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WatcherDaemon")

class WatcherService:
    def __init__(self, config_path: str, mqtt_host: str = "127.0.0.1", mqtt_port: int = 1883):
        self.engine = DynamicSeverityEngine(config_path)
        self.tracker = SimpleObjectTracker(iou_threshold=0.3)
        self.publisher = MQTTPublisher(host=mqtt_host, port=mqtt_port)
        self.actuator = PLCActuator()
        self.audit_logger = AuditLogger()
        self.frame_buffer = RollingFrameBuffer(fps=30, window_seconds=10)
        self.notifier = NotificationDispatcher()
        logger.info("Watcher Safety Service initialized with Persistent Object Tracker.")

    def ingest_frame(self, frame: np.ndarray):
        self.frame_buffer.push_frame(frame)

    def handle_frame_detections(
        self,
        sensor_id: str,
        zone_id: str,
        raw_detections: List[Dict[str, Any]]
    ):
        # 1. Update Persistent Tracker
        tracked_detections = self.tracker.update(raw_detections)

        # 2. Process Detections in Severity & Debouncing Engine
        for det in tracked_detections:
            class_name = det.get("class_name")
            confidence = det.get("confidence", 0.0)
            bbox = det.get("bounding_box", [])
            tracking_id = det.get("tracking_id")

            rule = self.engine.process_detection(class_name, confidence, tracking_id)
            if rule:
                payload = {
                    "sensor_id": sensor_id,
                    "zone_id": zone_id,
                    "tracking_id": tracking_id,
                    "event_type": rule.display_name,
                    "confidence": confidence,
                    "severity": rule.severity,
                    "action_type": rule.action_type
                }

                # Audit, Alerts, Actuation, Notifications
                self.audit_logger.log_incident(payload)
                if rule.severity == "CRITICAL":
                    self.frame_buffer.dump_mp4_clip()

                self.publisher.publish_event(
                    rule=rule,
                    sensor_id=sensor_id,
                    zone_id=zone_id,
                    confidence=confidence,
                    bounding_box=bbox,
                    tracking_id=tracking_id
                )

                if rule.action_type == "PLC_RELAY_TRIP" and rule.plc_config:
                    self.actuator.trigger_relay(rule.plc_config)

                self.notifier.dispatch_secondary_notifications(payload)
                self.engine.reset_tracker(tracking_id, class_name)

    def shutdown(self):
        self.publisher.disconnect()
        logger.info("Watcher Safety Service shut down.")
        
if __name__ == "__main__":
    config_path = os.getenv("CONFIG_PATH", "/app/config/watcher_stress_test_rules.yaml")
    mqtt_host = os.getenv("MQTT_HOST", "ccvnn-mqtt-broker")
    mqtt_port = int(os.getenv("MQTT_PORT", 1883))

    service = WatcherService(config_path=config_path, mqtt_host=mqtt_host, mqtt_port=mqtt_port)
    logger.info("Watcher Safety Service daemon running...")

    stop_requested = False

    def handle_sig(sig, frame):
        global stop_requested
        logger.info("Shutdown signal received. Exiting...")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        while not stop_requested:
            time.sleep(1)
    finally:
        service.shutdown()
        sys.exit(0)