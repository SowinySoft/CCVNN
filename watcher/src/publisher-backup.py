import json
import logging
import time
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from typing import Dict, Any, Optional
from watcher.src.engine import SeverityRule
# Add import at top of watcher/src/publisher.py:
from watcher.src.storage import OfflineStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MQTTPublisher")

class MQTTPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 1883):
        # Inside MQTTPublisher.__init__:
        self.storage = OfflineStorage()
        self.host = host
        self.port = port
        self.client = mqtt.Client(CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
        self._connect()

    def _connect(self):
        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            logger.info(f"Connected to MQTT broker at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")

    # Update publish_event method in watcher/src/publisher.py:
    def publish_event(
        self,
        rule: SeverityRule,
        sensor_id: str,
        zone_id: str,
        confidence: float,
        bounding_box: list,
        tracking_id: str
    ) -> bool:
        payload = {
            "event_id": f"evt_{int(time.time() * 1000)}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sensor_id": sensor_id,
            "zone_id": zone_id,
            "tracking_id": tracking_id,
            "event_type": rule.display_name,
            "severity": rule.severity,
            "confidence": round(confidence, 4),
            "bounding_box": bounding_box,
            "action_type": rule.action_type,
            "plc_config": rule.plc_config
        }

        target_topics = rule.topics if rule.topics else [f"factory/{zone_id}/alerts/{rule.severity.lower()}"]

        for topic in target_topics:
            if self.client.is_connected():
                res = self.client.publish(topic, json.dumps(payload), qos=1)
                if res.rc == mqtt.MQTT_ERR_SUCCESS:
                    logger.info(f"Published alert to topic [{topic}]: {payload['event_id']}")
                    self.flush_offline_buffer()
                else:
                    self.storage.buffer_event(topic, payload)
            else:
                self.storage.buffer_event(topic, payload)

        return True

    def flush_offline_buffer(self):
        """Re-flushes cached events when broker connection is online."""
        pending = self.storage.get_pending_events()
        if not pending or not self.client.is_connected():
            return

        logger.info(f"Flushing {len(pending)} cached offline events from SQLite...")
        for row_id, topic, payload in pending:
            res = self.client.publish(topic, json.dumps(payload), qos=1)
            if res.rc == mqtt.MQTT_ERR_SUCCESS:
                self.storage.remove_event(row_id)
                
    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()