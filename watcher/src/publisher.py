import sqlite3
import json
import logging
import time
from typing import Dict, Any, Optional

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

from paho.mqtt.enums import CallbackAPIVersion

from watcher.src.engine import SeverityRule
# Add import at top of watcher/src/publisher.py:
from watcher.src.storage import OfflineStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MQTTPublisher")

class MQTTPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 1883, db_path: str = "watcher_offline_queue.db"):
        # Inside MQTTPublisher.__init__:
        self.storage = OfflineStorage()
        self.host = host
        self.port = port
        self.db_path = db_path
        self._init_db()
        self.client = mqtt.Client(CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
        self._connect()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS offline_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            
    def _buffer_message(self, topic: str, payload: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO offline_messages (topic, payload) VALUES (?, ?)",
                (topic, json.dumps(payload))
            )
            conn.commit()

    def publish_event_dict(self, payload: Dict[str, Any], topic: str = "watcher/safety/alerts"):
        """Attempts MQTT delivery; falls back to SQLite buffer on connection failure."""
        try:
            if mqtt is None:
                raise ConnectionError("paho-mqtt library not available")

            client = mqtt.Client()
            client.connect(self.host, self.port, keepalive=2)
            client.publish(topic, json.dumps(payload), qos=1)
            client.disconnect()
            logger.info(f"Published MQTT event directly to {topic}")
        except Exception as e:
            logger.warning(f"MQTT publish failed ({e}). Buffering message to SQLite.")
            self._buffer_message(topic, payload)

    def reconnect_and_flush(self, host: Optional[str] = None, port: Optional[int] = None):
        """Flushes all queued SQLite messages once broker connectivity is restored."""
        if host:
            self.host = host
        if port:
            self.port = port

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, topic, payload FROM offline_messages ORDER BY id ASC")
            rows = cursor.fetchall()

            if not rows:
                return

            flushed_ids = []
            for row_id, topic, payload_str in rows:
                try:
                    payload = json.loads(payload_str)
                    # Send via restored host/port
                    if mqtt:
                        client = mqtt.Client()
                        client.connect(self.host, self.port, keepalive=2)
                        client.publish(topic, json.dumps(payload), qos=1)
                        client.disconnect()
                    flushed_ids.append(row_id)
                except Exception as e:
                    logger.error(f"Failed to flush queued message ID {row_id}: {e}")
                    break

            if flushed_ids:
                cursor.executemany("DELETE FROM offline_messages WHERE id = ?", [(i,) for i in flushed_ids])
                conn.commit()
                logger.info(f"Flushed {len(flushed_ids)} offline messages from SQLite queue.")

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